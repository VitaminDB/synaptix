//! Fused GroupNorm: `y = ((x - mean)/sqrt(var+eps)) * gamma + beta`, mean/var по
//! (per_group каналов × HW) на каждую пару (batch, group). One block per (b,g),
//! F32-аккумулятор. Заменяет ~12 decomposed-ops + медленный multi-dim reduce
//! (узкое место VAE/UNet на больших spatial). `has_affine=false` → без gamma/beta.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GroupNormParams {
    pub c: i32,
    pub hw: i32,
    pub g: i32,
    pub eps: f32,
    pub has_affine: i32,
    pub apply_silu: i32,
    pub x_offset: i64,
    pub w_offset: i64,
    pub b_offset: i64,
    pub y_offset: i64,
}
unsafe impl DeviceRepr for GroupNormParams {}

pub struct GroupNormKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
    f32_nhwc: CudaFunction,
    f16_nhwc: CudaFunction,
    bf16_nhwc: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GroupNormKernels>)>>> = OnceLock::new();

impl GroupNormKernels {
    pub fn for_context(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return Ok(v.clone());
                }
            }
        }
        let src = include_str!("../cu/fused/norm/group_norm.cu");
        let module = compile_module(ctx, src, "group_norm.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "group_norm_f32")?,
            f16: load_fn(&module, "group_norm_f16")?,
            bf16: load_fn(&module, "group_norm_bf16")?,
            f32_nhwc: load_fn(&module, "group_norm_nhwc_f32")?,
            f16_nhwc: load_fn(&module, "group_norm_nhwc_f16")?,
            bf16_nhwc: load_fn(&module, "group_norm_nhwc_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// GroupNorm из untyped `u8`-storage (для `Backend::group_norm`). `x`/`y` —
/// `[B,C,HW]` (логически), `w`/`bias` — `[C]`. Все в `dtype` (F32/F16/BF16) с
/// byte-offset'ами. `bias` без `weight` не поддержан (передавай оба или ни одного).
#[allow(clippy::too_many_arguments)]
pub fn run_u8(
    kernels: &GroupNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    affine: Option<(&CudaSlice<u8>, usize, &CudaSlice<u8>, usize)>,
    y: &mut CudaSlice<u8>,
    y_off: usize,
    b: u32,
    c: u32,
    hw: u32,
    g: u32,
    eps: f32,
    apply_silu: bool,
    nhwc: bool,
    dtype: DType,
) -> Result<()> {
    if b == 0 || c == 0 || hw == 0 || g == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (b as usize) * (c as usize) * (hw as usize);
    let cn = c as usize;
    let params = GroupNormParams {
        c: c as i32,
        hw: hw as i32,
        g: g as i32,
        eps,
        has_affine: if affine.is_some() { 1 } else { 0 },
        apply_silu: if apply_silu { 1 } else { 0 },
        x_offset: 0,
        w_offset: 0,
        b_offset: 0,
        y_offset: 0,
    };
    let cfg = LaunchConfig {
        grid_dim: (b * g, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("group_norm: transmute x".into()))?
            };
            let aff = match affine {
                Some((w, w_off, bb, b_off)) => {
                    let w_v = unsafe {
                        w.slice(w_off..w_off + cn * esz)
                            .transmute::<$t>(cn)
                            .ok_or_else(|| SynaptixError::Cuda("group_norm: transmute w".into()))?
                    };
                    let b_v = unsafe {
                        bb.slice(b_off..b_off + cn * esz)
                            .transmute::<$t>(cn)
                            .ok_or_else(|| SynaptixError::Cuda("group_norm: transmute b".into()))?
                    };
                    Some((w_v, b_v))
                }
                None => None,
            };
            let mut y_s = y.slice_mut(y_off..y_off + xn * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("group_norm: transmute y".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&x_v);
            match &aff {
                Some((w_v, b_v)) => {
                    bld.arg(w_v).arg(b_v);
                }
                None => {
                    bld.arg(&x_v).arg(&x_v);
                }
            };
            bld.arg(&mut y_v).arg(&params);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch group_norm: {e:?}")))?;
            }
        }};
    }
    match (dtype, nhwc) {
        (DType::F32, false) => go!(f32, &kernels.f32),
        (DType::F16, false) => go!(f16, &kernels.f16),
        (DType::BF16, false) => go!(bf16, &kernels.bf16),
        (DType::F32, true) => go!(f32, &kernels.f32_nhwc),
        (DType::F16, true) => go!(f16, &kernels.f16_nhwc),
        (DType::BF16, true) => go!(bf16, &kernels.bf16_nhwc),
        _ => return Err(SynaptixError::Unsupported("group_norm_u8: dtype")),
    }
    Ok(())
}

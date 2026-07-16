//! Fused PixelNorm (+опц. silu): `y = x / sqrt(mean_c(x²)+eps)` per-location по
//! каналам NCHW. Один kernel вместо decomposed-цепочки cast(f32)→sqr→mean→sqrt→
//! div→cast(+silu): ~21GB f32-трафика → 3 прохода bf16 (28% времени VAE-декода).
//! F32-аккумулятор; warp читает соседние локации (коалесцентно), цикл по C.

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
pub struct PixelNormParams {
    pub c: i32,
    pub s: i64,
    pub eps: f32,
    pub apply_silu: i32,
}
unsafe impl DeviceRepr for PixelNormParams {}

pub struct PixelNormKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<PixelNormKernels>)>>> = OnceLock::new();

impl PixelNormKernels {
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
        let src = include_str!("../cu/fused/norm/pixel_norm.cu");
        let module = compile_module(ctx, src, "pixel_norm.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "pixel_norm_f32")?,
            f16: load_fn(&module, "pixel_norm_f16")?,
            bf16: load_fn(&module, "pixel_norm_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// PixelNorm из untyped `u8`-storage: `x`/`y` — логически `[B,C,S]` contiguous,
/// S = D·H·W. VEC-хвост обрабатывается скалярно внутри ядра.
#[allow(clippy::too_many_arguments)]
pub fn run_u8(
    kernels: &PixelNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    y: &mut CudaSlice<u8>,
    y_off: usize,
    b: u32,
    c: u32,
    s: u64,
    eps: f32,
    apply_silu: bool,
    dtype: DType,
) -> Result<()> {
    if b == 0 || c == 0 || s == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (b as usize) * (c as usize) * (s as usize);
    let params = PixelNormParams {
        c: c as i32,
        s: s as i64,
        eps,
        apply_silu: if apply_silu { 1 } else { 0 },
    };
    let vec_n: u64 = if dtype == DType::F32 { 4 } else { 8 };
    let grid_x = s.div_ceil(vec_n * BLOCK as u64) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_x, b, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("pixel_norm: transmute x".into()))?
            };
            let mut y_s = y.slice_mut(y_off..y_off + xn * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("pixel_norm: transmute y".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&x_v).arg(&mut y_v).arg(&params);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch pixel_norm: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        _ => return Err(SynaptixError::Unsupported("pixel_norm_u8: dtype")),
    }
    Ok(())
}

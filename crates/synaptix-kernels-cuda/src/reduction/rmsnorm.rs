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
pub struct RmsNormParams {
    pub batch: i32,
    pub hidden: i32,
    pub eps: f32,
    pub variant: i32,
    pub x_offset: i64,
    pub w_offset: i64,
    pub g_offset: i64,
    pub y_offset: i64,
    pub x_row_stride: i64,
    pub g_row_stride: i64,
    pub y_row_stride: i64,
}
unsafe impl DeviceRepr for RmsNormParams {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RmsVariant {
    Plain,
    Qwen,
}

impl RmsVariant {
    fn as_i32(self) -> i32 {
        match self {
            Self::Plain => 0,
            Self::Qwen => 1,
        }
    }
}

pub struct RmsNormKernels {
    _module: Arc<CudaModule>,
    rms_norm_f32: CudaFunction,
    rms_norm_f16: CudaFunction,
    rms_norm_bf16: CudaFunction,
    rms_norm_gated_f32: CudaFunction,
    rms_norm_gated_f16: CudaFunction,
    rms_norm_gated_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<RmsNormKernels>)>>> = OnceLock::new();

impl RmsNormKernels {
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
        let src = include_str!("../cu/reduction/rms_norm.cu");
        let module = compile_module(ctx, src, "rms_norm.cu")?;
        let new = Arc::new(Self {
            rms_norm_f32: load_fn(&module, "rms_norm_f32")?,
            rms_norm_f16: load_fn(&module, "rms_norm_f16")?,
            rms_norm_bf16: load_fn(&module, "rms_norm_bf16")?,
            rms_norm_gated_f32: load_fn(&module, "rms_norm_gated_f32")?,
            rms_norm_gated_f16: load_fn(&module, "rms_norm_gated_f16")?,
            rms_norm_gated_bf16: load_fn(&module, "rms_norm_gated_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn launch_cfg(batch: u32, hidden: u32) -> LaunchConfig {
    // Decode (M≤8): grid=batch → один блок на строку, мало варпов на SM (BLOCK=256
    // = 8 варпов) → латентность памяти не скрывается (ncu: M=1 norm ~18us при
    // dram 0.2%). Даём до 1024 нитей (32 варпа = потолок warp_sums[32]) → MLP
    // выше, латентность спрятана. Prefill (большой batch): блоков уже grid=batch,
    // SM насыщены → оставляем 256 (не трогаем профиль prefill / bit-exact).
    let block = if batch <= 8 {
        hidden.next_multiple_of(32).clamp(BLOCK, 1024)
    } else {
        BLOCK
    };
    LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn build_params(
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
    gated: bool,
) -> RmsNormParams {
    RmsNormParams {
        batch: batch as i32,
        hidden: hidden as i32,
        eps,
        variant: variant.as_i32(),
        x_offset: 0,
        w_offset: 0,
        g_offset: 0,
        y_offset: 0,
        x_row_stride: hidden as i64,
        g_row_stride: if gated { hidden as i64 } else { 0 },
        y_row_stride: hidden as i64,
    }
}

pub fn run_rms_norm<T: DeviceRepr>(
    kernels: &RmsNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    w: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.rms_norm_f32,
        DType::F16 => &kernels.rms_norm_f16,
        DType::BF16 => &kernels.rms_norm_bf16,
        _ => {
            return Err(SynaptixError::Unsupported(
                "rms_norm: dtype must be F32/F16/BF16",
            ))
        }
    };
    let params = build_params(batch, hidden, eps, variant, false);
    let cfg = launch_cfg(batch, hidden);
    let mut b = stream.launch_builder(func);
    b.arg(x).arg(w).arg(y).arg(&params);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch rms_norm: {e:?}")))?;
    }
    Ok(())
}

/// Fused RMSNorm из untyped `u8`-storage (для `Backend::rms_norm`). Нормирует
/// по last dim: `x[batch, hidden]`, `w[hidden]`, `y[batch, hidden]`. Один launch
/// на batch строк, F32-аккумулятор внутри. Принимает byte-offset'ы (view режется).
#[allow(clippy::too_many_arguments)]
pub fn run_rms_norm_u8(
    kernels: &RmsNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    w: &CudaSlice<u8>,
    w_off: usize,
    y: &mut CudaSlice<u8>,
    y_off: usize,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
    dtype: DType,
) -> Result<()> {
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (batch as usize) * (hidden as usize);
    let wn = hidden as usize;
    let params = build_params(batch, hidden, eps, variant, false);
    let cfg = launch_cfg(batch, hidden);

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rms_norm: transmute x".into()))?
            };
            let w_v = unsafe {
                w.slice(w_off..w_off + wn * esz)
                    .transmute::<$t>(wn)
                    .ok_or_else(|| SynaptixError::Cuda("rms_norm: transmute w".into()))?
            };
            let mut y_s = y.slice_mut(y_off..y_off + xn * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rms_norm: transmute y".into()))?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&x_v).arg(&w_v).arg(&mut y_v).arg(&params);
            unsafe {
                b.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch rms_norm_u8: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::F32 => go!(f32, &kernels.rms_norm_f32),
        DType::F16 => go!(f16, &kernels.rms_norm_f16),
        DType::BF16 => go!(bf16, &kernels.rms_norm_bf16),
        _ => return Err(SynaptixError::Unsupported("rms_norm_u8: dtype")),
    }
    Ok(())
}

pub fn run_rms_norm_gated<T: DeviceRepr>(
    kernels: &RmsNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    gate: &CudaSlice<T>,
    w: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.rms_norm_gated_f32,
        DType::F16 => &kernels.rms_norm_gated_f16,
        DType::BF16 => &kernels.rms_norm_gated_bf16,
        _ => {
            return Err(SynaptixError::Unsupported(
                "rms_norm_gated: dtype must be F32/F16/BF16",
            ))
        }
    };
    let params = build_params(batch, hidden, eps, RmsVariant::Plain, true);
    let cfg = launch_cfg(batch, hidden);
    let mut b = stream.launch_builder(func);
    b.arg(x).arg(gate).arg(w).arg(y).arg(&params);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch rms_norm_gated: {e:?}")))?;
    }
    Ok(())
}

//! LayerNorm forward: `y = ((x - mean) / sqrt(var + eps)) * gamma + beta`.
//!
//! Один CUDA block per row. F32/F16/BF16. F32 accumulator для mixed-precision.
//! Bias (beta) опциональный: при `has_beta=false` норм-выход умножается на
//! gamma без сдвига.

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
pub struct LayerNormParams {
    pub batch: i32,
    pub hidden: i32,
    pub eps: f32,
    pub has_beta: i32,
    pub x_offset: i64,
    pub w_offset: i64,
    pub b_offset: i64,
    pub y_offset: i64,
    pub x_row_stride: i64,
    pub y_row_stride: i64,
}
unsafe impl DeviceRepr for LayerNormParams {}

pub struct LayerNormKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<LayerNormKernels>)>>> = OnceLock::new();

impl LayerNormKernels {
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
        let src = include_str!("../cu/reduction/layernorm.cu");
        let module = compile_module(ctx, src, "layernorm.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "layernorm_f32")?,
            f16: load_fn(&module, "layernorm_f16")?,
            bf16: load_fn(&module, "layernorm_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn build_params(batch: u32, hidden: u32, eps: f32, has_beta: bool) -> LayerNormParams {
    LayerNormParams {
        batch: batch as i32,
        hidden: hidden as i32,
        eps,
        has_beta: if has_beta { 1 } else { 0 },
        x_offset: 0,
        w_offset: 0,
        b_offset: 0,
        y_offset: 0,
        x_row_stride: hidden as i64,
        y_row_stride: hidden as i64,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run<T: DeviceRepr>(
    kernels: &LayerNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    w: &CudaSlice<T>,
    beta: Option<&CudaSlice<T>>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        _ => {
            return Err(SynaptixError::Unsupported(
                "layernorm: dtype must be F32/F16/BF16",
            ))
        }
    };
    let params = build_params(batch, hidden, eps, beta.is_some());
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut b = stream.launch_builder(func);
    b.arg(x).arg(w);
    if let Some(beta_buf) = beta {
        b.arg(beta_buf);
    } else {
        b.arg(w);
    }
    b.arg(&mut *y).arg(&params);
    unsafe {
        b.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch layernorm: {e:?}")))?;
    }
    Ok(())
}

/// LayerNorm из untyped `u8`-storage (для `Backend::layer_norm`). `x`/`w`/`beta`/`y`
/// — `dtype` (F32/F16/BF16) с byte-offset'ами. `beta` опциональный (None → без сдвига).
#[allow(clippy::too_many_arguments)]
pub fn run_u8(
    kernels: &LayerNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    x_off: usize,
    w: &CudaSlice<u8>,
    w_off: usize,
    beta: Option<(&CudaSlice<u8>, usize)>,
    y: &mut CudaSlice<u8>,
    y_off: usize,
    batch: u32,
    hidden: u32,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    let xn = (batch as usize) * (hidden as usize);
    let wn = hidden as usize;
    let params = build_params(batch, hidden, eps, beta.is_some());
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let x_v = unsafe {
                x.slice(x_off..x_off + xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("layernorm: transmute x".into()))?
            };
            let w_v = unsafe {
                w.slice(w_off..w_off + wn * esz)
                    .transmute::<$t>(wn)
                    .ok_or_else(|| SynaptixError::Cuda("layernorm: transmute w".into()))?
            };
            let beta_v = match beta {
                Some((bb, b_off)) => Some(unsafe {
                    bb.slice(b_off..b_off + wn * esz)
                        .transmute::<$t>(wn)
                        .ok_or_else(|| SynaptixError::Cuda("layernorm: transmute beta".into()))?
                }),
                None => None,
            };
            let mut y_s = y.slice_mut(y_off..y_off + xn * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("layernorm: transmute y".into()))?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&x_v).arg(&w_v);
            match &beta_v {
                Some(bv) => b.arg(bv),
                None => b.arg(&w_v),
            };
            b.arg(&mut y_v).arg(&params);
            unsafe {
                b.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch layernorm_u8: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        _ => return Err(SynaptixError::Unsupported("layernorm_u8: dtype")),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_f32(
    kernels: &LayerNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    w: &CudaSlice<f32>,
    beta: Option<&CudaSlice<f32>>,
    y: &mut CudaSlice<f32>,
    batch: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    run::<f32>(
        kernels,
        stream,
        x,
        w,
        beta,
        y,
        batch,
        hidden,
        eps,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_f16(
    kernels: &LayerNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    w: &CudaSlice<f16>,
    beta: Option<&CudaSlice<f16>>,
    y: &mut CudaSlice<f16>,
    batch: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    run::<f16>(
        kernels,
        stream,
        x,
        w,
        beta,
        y,
        batch,
        hidden,
        eps,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_bf16(
    kernels: &LayerNormKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    w: &CudaSlice<bf16>,
    beta: Option<&CudaSlice<bf16>>,
    y: &mut CudaSlice<bf16>,
    batch: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    run::<bf16>(
        kernels,
        stream,
        x,
        w,
        beta,
        y,
        batch,
        hidden,
        eps,
        DType::BF16,
    )
}

//! Fused RMSNorm + Residual: `residual = x + residual; y = rmsnorm(residual) * weight`.
//!
//! Заменяет 2 kernel calls (add residual + rms_norm) на один. Экономит один
//! полный memory pass по hidden buffer на каждом transformer block residual.

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
use crate::reduction::rmsnorm::RmsVariant;

const BLOCK: u32 = 256;

pub struct RmsNormResidualKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
    split_f32: CudaFunction,
    split_f16: CudaFunction,
    split_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<RmsNormResidualKernels>)>>> = OnceLock::new();

impl RmsNormResidualKernels {
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
        let src = include_str!("../cu/fused/norm/rmsnorm_residual.cu");
        let module = compile_module(ctx, src, "rmsnorm_residual.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "rmsnorm_residual_f32")?,
            f16: load_fn(&module, "rmsnorm_residual_f16")?,
            bf16: load_fn(&module, "rmsnorm_residual_bf16")?,
            split_f32: load_fn(&module, "rmsnorm_residual_split_f32")?,
            split_f16: load_fn(&module, "rmsnorm_residual_split_f16")?,
            split_bf16: load_fn(&module, "rmsnorm_residual_split_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run<T: DeviceRepr>(
    kernels: &RmsNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    residual: &mut CudaSlice<T>,
    weight: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "rmsnorm_residual: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i = batch as i32;
    let hidden_i = hidden as i32;
    let qwen: i32 = match variant {
        RmsVariant::Plain => 0,
        RmsVariant::Qwen => 1,
    };
    let mut bld = stream.launch_builder(func);
    bld.arg(x)
        .arg(&mut *residual)
        .arg(weight)
        .arg(&mut *y)
        .arg(&batch_i)
        .arg(&hidden_i)
        .arg(&eps)
        .arg(&qwen);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch rmsnorm_residual: {e:?}")))?;
    }
    Ok(())
}

/// Out-of-place: `hidden_out = x + residual; y = RMSNorm(hidden_out)*weight`.
/// НЕ мутирует `residual` (для Tensor-семантики). Для малого batch (decode)
/// поднимаем block до 1024 нитей — латентность M=1 строки не скрывается при 256.
#[allow(clippy::too_many_arguments)]
pub fn run_split<T: DeviceRepr>(
    kernels: &RmsNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    residual: &CudaSlice<T>,
    weight: &CudaSlice<T>,
    hidden_out: &mut CudaSlice<T>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.split_f32,
        DType::F16 => &kernels.split_f16,
        DType::BF16 => &kernels.split_bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "rmsnorm_residual_split: unsupported dtype {other:?}"
            )))
        }
    };
    let block = if batch <= 8 {
        hidden.next_multiple_of(32).clamp(BLOCK, 1024)
    } else {
        BLOCK
    };
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i = batch as i32;
    let hidden_i = hidden as i32;
    let qwen: i32 = match variant {
        RmsVariant::Plain => 0,
        RmsVariant::Qwen => 1,
    };
    let mut bld = stream.launch_builder(func);
    bld.arg(x)
        .arg(residual)
        .arg(weight)
        .arg(&mut *hidden_out)
        .arg(&mut *y)
        .arg(&batch_i)
        .arg(&hidden_i)
        .arg(&eps)
        .arg(&qwen);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch rmsnorm_residual_split: {e:?}")))?;
    }
    Ok(())
}

/// Untyped u8-storage обёртка [`run_split`] (для `Backend::rms_norm_residual`).
/// Все буферы contiguous, byte-offset 0 (decode-тензоры свежие).
#[allow(clippy::too_many_arguments)]
pub fn run_split_u8(
    kernels: &RmsNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<u8>,
    residual: &CudaSlice<u8>,
    w: &CudaSlice<u8>,
    hidden_out: &mut CudaSlice<u8>,
    y: &mut CudaSlice<u8>,
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
    let func = match dtype {
        DType::F32 => &kernels.split_f32,
        DType::F16 => &kernels.split_f16,
        DType::BF16 => &kernels.split_bf16,
        _ => return Err(SynaptixError::Unsupported("rmsnorm_residual_split_u8: dtype")),
    };
    let block = if batch <= 8 {
        hidden.next_multiple_of(32).clamp(BLOCK, 1024)
    } else {
        BLOCK
    };
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i = batch as i32;
    let hidden_i = hidden as i32;
    let qwen: i32 = match variant {
        RmsVariant::Plain => 0,
        RmsVariant::Qwen => 1,
    };
    macro_rules! go {
        ($t:ty) => {{
            let x_v = unsafe {
                x.slice(0..xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rmsnorm_residual_split: transmute x".into()))?
            };
            let r_v = unsafe {
                residual
                    .slice(0..xn * esz)
                    .transmute::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rmsnorm_residual_split: transmute r".into()))?
            };
            let w_v = unsafe {
                w.slice(0..wn * esz)
                    .transmute::<$t>(wn)
                    .ok_or_else(|| SynaptixError::Cuda("rmsnorm_residual_split: transmute w".into()))?
            };
            let mut h_s = hidden_out.slice_mut(0..xn * esz);
            let mut h_v = unsafe {
                h_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rmsnorm_residual_split: transmute h".into()))?
            };
            let mut y_s = y.slice_mut(0..xn * esz);
            let mut y_v = unsafe {
                y_s.transmute_mut::<$t>(xn)
                    .ok_or_else(|| SynaptixError::Cuda("rmsnorm_residual_split: transmute y".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&x_v)
                .arg(&r_v)
                .arg(&w_v)
                .arg(&mut h_v)
                .arg(&mut y_v)
                .arg(&batch_i)
                .arg(&hidden_i)
                .arg(&eps)
                .arg(&qwen);
            unsafe {
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch rmsnorm_residual_split_u8: {e:?}"))
                })?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32),
        DType::F16 => go!(f16),
        DType::BF16 => go!(bf16),
        _ => return Err(SynaptixError::Unsupported("rmsnorm_residual_split_u8: dtype")),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_f32(
    kernels: &RmsNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    residual: &mut CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
) -> Result<()> {
    run::<f32>(
        kernels,
        stream,
        x,
        residual,
        weight,
        y,
        batch,
        hidden,
        eps,
        variant,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_f16(
    kernels: &RmsNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    residual: &mut CudaSlice<f16>,
    weight: &CudaSlice<f16>,
    y: &mut CudaSlice<f16>,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
) -> Result<()> {
    run::<f16>(
        kernels,
        stream,
        x,
        residual,
        weight,
        y,
        batch,
        hidden,
        eps,
        variant,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_bf16(
    kernels: &RmsNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    residual: &mut CudaSlice<bf16>,
    weight: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    batch: u32,
    hidden: u32,
    eps: f32,
    variant: RmsVariant,
) -> Result<()> {
    run::<bf16>(
        kernels,
        stream,
        x,
        residual,
        weight,
        y,
        batch,
        hidden,
        eps,
        variant,
        DType::BF16,
    )
}

//! Fused: `residual = x + residual; y = LayerNorm(residual) * gamma + beta`.
//!
//! Полный LayerNorm (mean + var), F32/F16/BF16, f32-аккумулятор. Один launch
//! вместо residual-add + layernorm — экономит memory pass по hidden buffer
//! (аналог [`crate::fused::rmsnorm_residual`]). `beta` опционален.

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

pub struct LayerNormResidualKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<LayerNormResidualKernels>)>>> = OnceLock::new();

impl LayerNormResidualKernels {
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
        let src = include_str!("../cu/fused/norm/layernorm_residual.cu");
        let module = compile_module(ctx, src, "layernorm_residual.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "layernorm_residual_f32")?,
            f16: load_fn(&module, "layernorm_residual_f16")?,
            bf16: load_fn(&module, "layernorm_residual_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `residual = x + residual; y = ((residual - mean) / sqrt(var + eps)) * gamma + beta`.
/// `x`, `residual`, `y` — (batch, hidden) row-major; `gamma`/`beta` — (hidden,).
/// `beta == None` ⟹ без сдвига.
#[allow(clippy::too_many_arguments)]
pub fn layernorm_residual<T: DeviceRepr>(
    kernels: &LayerNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    residual: &mut CudaSlice<T>,
    gamma: &CudaSlice<T>,
    beta: Option<&CudaSlice<T>>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    eps: f32,
    dtype: DType,
) -> Result<()> {
    if batch == 0 || hidden == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "layernorm_residual: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (batch, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i = batch as i32;
    let hidden_i = hidden as i32;
    let has_beta_i: i32 = if beta.is_some() { 1 } else { 0 };
    // beta может быть None — kernel читает его только при has_beta==1; передаём
    // gamma как валидный placeholder-pointer.
    let beta_ptr = beta.unwrap_or(gamma);
    let mut bld = stream.launch_builder(func);
    bld.arg(x)
        .arg(&mut *residual)
        .arg(gamma)
        .arg(beta_ptr)
        .arg(&has_beta_i)
        .arg(&mut *y)
        .arg(&batch_i)
        .arg(&hidden_i)
        .arg(&eps);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch layernorm_residual: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn layernorm_residual_f32(
    kernels: &LayerNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    residual: &mut CudaSlice<f32>,
    gamma: &CudaSlice<f32>,
    beta: Option<&CudaSlice<f32>>,
    y: &mut CudaSlice<f32>,
    batch: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    layernorm_residual::<f32>(
        kernels,
        stream,
        x,
        residual,
        gamma,
        beta,
        y,
        batch,
        hidden,
        eps,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layernorm_residual_f16(
    kernels: &LayerNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    residual: &mut CudaSlice<f16>,
    gamma: &CudaSlice<f16>,
    beta: Option<&CudaSlice<f16>>,
    y: &mut CudaSlice<f16>,
    batch: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    layernorm_residual::<f16>(
        kernels,
        stream,
        x,
        residual,
        gamma,
        beta,
        y,
        batch,
        hidden,
        eps,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn layernorm_residual_bf16(
    kernels: &LayerNormResidualKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    residual: &mut CudaSlice<bf16>,
    gamma: &CudaSlice<bf16>,
    beta: Option<&CudaSlice<bf16>>,
    y: &mut CudaSlice<bf16>,
    batch: u32,
    hidden: u32,
    eps: f32,
) -> Result<()> {
    layernorm_residual::<bf16>(
        kernels,
        stream,
        x,
        residual,
        gamma,
        beta,
        y,
        batch,
        hidden,
        eps,
        DType::BF16,
    )
}

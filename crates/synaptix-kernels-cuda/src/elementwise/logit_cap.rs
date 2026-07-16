//! Logit / attention soft-cap: `out = cap * tanh(x / cap)` (Gemma2/Gemma3).
//!
//! Elementwise, f32-аккумулятор. F32/F16/BF16. Поддерживает in-place (`x == out`).
//! Семантика совпадает с `synaptix_ops::norm::soft_cap`.

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

pub struct LogitCapKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<LogitCapKernels>)>>> = OnceLock::new();

impl LogitCapKernels {
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
        let src = include_str!("../cu/elementwise/logit_cap.cu");
        let module = compile_module(ctx, src, "logit_cap.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "logit_cap_f32")?,
            f16: load_fn(&module, "logit_cap_f16")?,
            bf16: load_fn(&module, "logit_cap_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `out = cap * tanh(x / cap)` поэлементно. `cap` должен быть ненулевым.
pub fn logit_cap<T: DeviceRepr>(
    kernels: &LogitCapKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    out: &mut CudaSlice<T>,
    cap: f32,
    n: u32,
    dtype: DType,
) -> Result<()> {
    if cap == 0.0 {
        return Err(SynaptixError::Cuda(
            "logit_cap: cap должен быть ненулевым".to_string(),
        ));
    }
    if n == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "logit_cap: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (n.div_ceil(BLOCK), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_i = n as i32;
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(&mut *out).arg(&cap).arg(&n_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch logit_cap: {e:?}")))?;
    }
    Ok(())
}

pub fn logit_cap_f32(
    kernels: &LogitCapKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    cap: f32,
    n: u32,
) -> Result<()> {
    logit_cap::<f32>(kernels, stream, x, out, cap, n, DType::F32)
}

pub fn logit_cap_f16(
    kernels: &LogitCapKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    cap: f32,
    n: u32,
) -> Result<()> {
    logit_cap::<f16>(kernels, stream, x, out, cap, n, DType::F16)
}

pub fn logit_cap_bf16(
    kernels: &LogitCapKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    cap: f32,
    n: u32,
) -> Result<()> {
    logit_cap::<bf16>(kernels, stream, x, out, cap, n, DType::BF16)
}

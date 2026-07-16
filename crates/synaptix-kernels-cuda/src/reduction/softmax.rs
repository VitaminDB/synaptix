//! Numerically stable softmax по последнему dim.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;

pub struct SoftmaxKernels {
    _module: Arc<CudaModule>,
    softmax_f32: CudaFunction,
    softmax_f16: CudaFunction,
    softmax_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<SoftmaxKernels>)>>> = OnceLock::new();

impl SoftmaxKernels {
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
        let src = include_str!("../cu/reduction/softmax.cu");
        let module = compile_module(ctx, src, "softmax.cu")?;
        let new = Arc::new(Self {
            softmax_f32: load_fn(&module, "softmax_f32")?,
            softmax_f16: load_fn(&module, "softmax_f16")?,
            softmax_bf16: load_fn(&module, "softmax_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

pub fn run_softmax<T: DeviceRepr>(
    kernels: &SoftmaxKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    batch: u32,
    hidden: u32,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.softmax_f32,
        DType::F16 => &kernels.softmax_f16,
        DType::BF16 => &kernels.softmax_bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "softmax: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let hidden_i = hidden as i32;
    let batch_i = batch as i32;
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(&mut *y).arg(&hidden_i).arg(&batch_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch softmax: {e:?}")))?;
    }
    Ok(())
}

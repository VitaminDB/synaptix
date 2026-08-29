//! Top-k по широким строкам: выбор блоков индексатором QSA.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;

pub struct TopkWideKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<TopkWideKernels>)>>> = OnceLock::new();

impl TopkWideKernels {
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
        let src = include_str!("../cu/fused/topk_wide.cu");
        let module = compile_module(ctx, src, "topk_wide.cu")?;
        let new = Arc::new(Self { f32: load_fn(&module, "topk_wide_f32")?, _module: module });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

pub fn topk_wide_f32(
    kernels: &TopkWideKernels,
    stream: &Arc<CudaStream>,
    scores: &CudaSlice<u8>,
    valid: &CudaSlice<u8>,
    out: &mut CudaSlice<u8>,
    rows: u32,
    cols: u32,
    k: u32,
) -> Result<()> {
    if rows == 0 || k == 0 {
        return Ok(());
    }
    if k > cols {
        return Err(SynaptixError::Unsupported("topk_wide: k шире строки"));
    }
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.f32);
    bld.arg(scores).arg(valid).arg(out).arg(&rows).arg(&cols).arg(&k);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch topk_wide: {e:?}")))?;
    }
    Ok(())
}

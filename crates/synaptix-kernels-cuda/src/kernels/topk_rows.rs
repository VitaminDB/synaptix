//! Top-k по строкам: выбор экспертов роутером MoE.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;
/// Строка целиком лежит в shared, поэтому ширина ограничена.
pub const MAX_COLS: usize = 2048;

pub struct TopkRowsKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<TopkRowsKernels>)>>> = OnceLock::new();

impl TopkRowsKernels {
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
        let src = include_str!("../cu/fused/topk_rows.cu");
        let module = compile_module(ctx, src, "topk_rows.cu")?;
        let new = Arc::new(Self { f32: load_fn(&module, "topk_rows_f32")?, _module: module });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn topk_rows_f32(
    kernels: &TopkRowsKernels,
    stream: &Arc<CudaStream>,
    scores: &CudaSlice<u8>,
    out_idx: &mut CudaSlice<u8>,
    out_val: &mut CudaSlice<u8>,
    rows: u32,
    cols: u32,
    k: u32,
) -> Result<()> {
    if rows == 0 || k == 0 {
        return Ok(());
    }
    if cols as usize > MAX_COLS || k > cols {
        return Err(SynaptixError::Unsupported("topk_rows: строка шире потолка"));
    }
    let smem = (cols + BLOCK) * 4 + BLOCK * 4;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let mut bld = stream.launch_builder(&kernels.f32);
    bld.arg(scores).arg(out_idx).arg(out_val).arg(&rows).arg(&cols).arg(&k);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch topk_rows: {e:?}")))?;
    }
    Ok(())
}

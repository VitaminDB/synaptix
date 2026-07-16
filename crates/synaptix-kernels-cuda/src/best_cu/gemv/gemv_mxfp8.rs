use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaViewMut, LaunchConfig,
    PushKernelArg,
};
use half::f16;
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const WARPS: u32 = 8;
const THREADS: u32 = WARPS * 32;

pub struct GemvMxfp8Kernels {
    _module: Arc<CudaModule>,
    gemv: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GemvMxfp8Kernels>)>>> = OnceLock::new();

impl GemvMxfp8Kernels {
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
        let src = include_str!("gemv_mxfp8.cu");
        let module = compile_module_with_opts(ctx, src, "gemv_mxfp8.cu", &[], Some("sm_120a"))?;
        let gemv = load_fn(&module, "gemv_mxfp8_e4m3")?;
        let new = Arc::new(Self {
            gemv,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

// MXFP8 GEMV (decode, M=1): y[N] = W[N,K] @ x[K]. w/x — E4M3 байты (natural
// [.,K]); sw/sx — E8M0 per-32-block scales (natural [., K/32]). out — f16 [N].
#[allow(clippy::too_many_arguments)]
pub fn gemv_mxfp8(
    kernels: &GemvMxfp8Kernels,
    stream: &Arc<CudaStream>,
    w: &CudaSlice<u8>,
    sw: &CudaSlice<u8>,
    x: &CudaSlice<u8>,
    sx: &CudaSlice<u8>,
    out: &mut CudaViewMut<f16>,
    n: u32,
    k: u32,
) -> Result<()> {
    if n == 0 {
        return Ok(());
    }
    let grid = n.div_ceil(WARPS);
    let (ni, ki) = (n as i32, k as i32);
    let mut bld = stream.launch_builder(&kernels.gemv);
    bld.arg(w)
        .arg(sw)
        .arg(x)
        .arg(sx)
        .arg(&mut *out)
        .arg(&ni)
        .arg(&ki);
    unsafe {
        bld.launch(LaunchConfig {
            grid_dim: (grid, 1, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        })
        .map_err(|e| SynaptixError::Cuda(format!("launch gemv_mxfp8: {e:?}")))?;
    }
    Ok(())
}

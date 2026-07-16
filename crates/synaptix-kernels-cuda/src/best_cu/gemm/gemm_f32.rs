use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const BM: u32 = 64;
const BN: u32 = 64;
const THREADS: u32 = 256;

pub struct GemmF32Kernels {
    _module: Arc<CudaModule>,
    nn: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GemmF32Kernels>)>>> = OnceLock::new();

impl GemmF32Kernels {
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
        let src = include_str!("gemm_f32.cu");
        let module = compile_module_with_opts(ctx, src, "gemm_f32.cu", &[], Some("sm_120a"))?;
        let nn = load_fn(&module, "gemm_f32_nn")?;
        let new = Arc::new(Self {
            nn,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

// F32 NN GEMM C[b,M,N] = A[b,M,K] @ B[(b|1),K,N] (истинный f32, SIMT). Любые
// M/N/K (bounds). batched: per-batch launch; b_broadcast → B-offset 0.
#[allow(clippy::too_many_arguments)]
pub fn gemm_f32_nn_u8(
    kernels: &GemmF32Kernels,
    stream: &Arc<CudaStream>,
    a: &CudaSlice<u8>,
    b: &CudaSlice<u8>,
    c: &mut CudaSlice<u8>,
    m: u32,
    n: u32,
    k: u32,
    batch: u32,
    b_broadcast: bool,
) -> Result<()> {
    if m == 0 || n == 0 || batch == 0 {
        return Ok(());
    }
    let (am, bk, cm) = ((m * k) as usize, (k * n) as usize, (m * n) as usize);
    let (mi, ni, ki) = (m as i32, n as i32, k as i32);
    for bi in 0..batch as usize {
        let a_off = bi * am * 4;
        let b_off = if b_broadcast { 0 } else { bi * bk * 4 };
        let c_off = bi * cm * 4;
        let a_v = unsafe { a.slice(a_off..a_off + am * 4).transmute::<f32>(am) }
            .ok_or_else(|| SynaptixError::Cuda("gemm_f32: transmute a".into()))?;
        let b_v = unsafe { b.slice(b_off..b_off + bk * 4).transmute::<f32>(bk) }
            .ok_or_else(|| SynaptixError::Cuda("gemm_f32: transmute b".into()))?;
        let mut c_s = c.slice_mut(c_off..c_off + cm * 4);
        let mut c_v = unsafe { c_s.transmute_mut::<f32>(cm) }
            .ok_or_else(|| SynaptixError::Cuda("gemm_f32: transmute c".into()))?;
        let launch = LaunchConfig {
            grid_dim: (n.div_ceil(BN), m.div_ceil(BM), 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut bld = stream.launch_builder(&kernels.nn);
        bld.arg(&a_v)
            .arg(&b_v)
            .arg(&mut c_v)
            .arg(&mi)
            .arg(&ni)
            .arg(&ki);
        unsafe {
            bld.launch(launch)
                .map_err(|e| SynaptixError::Cuda(format!("launch gemm_f32_nn: {e:?}")))?;
        }
    }
    Ok(())
}

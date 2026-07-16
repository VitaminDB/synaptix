//! Fused GEGLU split-activation: `h[T,I] = p[:,:I] * gelu_exact(p[:,I:])` за один
//! проход (вместо narrow×2 + contiguous×2 + gelu + mul). Для SDXL UNet FF.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct GegluSplitKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<GegluSplitKernels>)>>> = OnceLock::new();

impl GegluSplitKernels {
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
        let src = include_str!("../cu/fused/mlp/geglu_split.cu");
        let module = compile_module(ctx, src, "geglu_split.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "geglu_split_f32")?,
            f16: load_fn(&module, "geglu_split_f16")?,
            bf16: load_fn(&module, "geglu_split_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `inp` `[T, 2*inner]` → `out` `[T, inner]` (offset в БАЙТАХ, transmute по dtype).
#[allow(clippy::too_many_arguments)]
pub fn run_geglu_split_u8(
    kernels: &GegluSplitKernels,
    stream: &Arc<CudaStream>,
    inp: &CudaSlice<u8>,
    inp_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    t: u64,
    inner: u32,
    dtype: DType,
) -> Result<()> {
    let n_out = (t as usize) * (inner as usize);
    let n_in = n_out * 2;
    if n_out == 0 {
        return Ok(());
    }
    let esz = (dtype.size_in_bits() / 8) as usize;
    const BLOCK: u32 = 256;
    let grid = ((n_out as u64).div_ceil(BLOCK as u64).min(65535) as u32).max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let t_i = t as i64;
    let inner_i = inner as i32;

    macro_rules! go {
        ($ty:ty, $func:expr) => {{
            let in_v = unsafe {
                inp.slice(inp_off..inp_off + n_in * esz)
                    .transmute::<$ty>(n_in)
                    .ok_or_else(|| SynaptixError::Cuda("geglu_split: transmute in".into()))?
            };
            let mut out_s = out.slice_mut(out_off..out_off + n_out * esz);
            let mut out_v = unsafe {
                out_s
                    .transmute_mut::<$ty>(n_out)
                    .ok_or_else(|| SynaptixError::Cuda("geglu_split: transmute out".into()))?
            };
            let mut b = stream.launch_builder($func);
            b.arg(&in_v).arg(&mut out_v).arg(&t_i).arg(&inner_i);
            unsafe {
                b.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch geglu_split: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        other => return Err(SynaptixError::Cuda(format!("geglu_split: dtype {other:?}"))),
    }
    Ok(())
}

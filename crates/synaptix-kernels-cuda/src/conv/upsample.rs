//! nearest-2x upsample CUDA kernel: in `[B,C,H,W]` → out `[B,C,2H,2W]`.
//! Заменяет cat-based upsample (тот на CUDA попадал в медленный strided-copy
//! путь — 19с на 256² в VAE-декодере). Один launch, memory-bound.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct Upsample2xKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<Upsample2xKernels>)>>> = OnceLock::new();

impl Upsample2xKernels {
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
        let src = include_str!("../cu/conv/upsample_nearest2x.cu");
        let module = compile_module(ctx, src, "upsample_nearest2x.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "upsample_nearest2x_f32")?,
            f16: load_fn(&module, "upsample_nearest2x_f16")?,
            bf16: load_fn(&module, "upsample_nearest2x_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// u8-вход (для Backend::upsample_nearest2x). `input` `[B,C,H,W]` → `output`
/// `[B,C,2H,2W]` (offset в БАЙТАХ), транзмутируется по `dtype`.
#[allow(clippy::too_many_arguments)]
pub fn run_upsample2x_u8(
    kernels: &Upsample2xKernels,
    stream: &Arc<CudaStream>,
    input: &CudaSlice<u8>,
    input_off: usize,
    output: &mut CudaSlice<u8>,
    output_off: usize,
    b: u32,
    c: u32,
    h: u32,
    w: u32,
    dtype: DType,
) -> Result<()> {
    let n_in = (b as usize) * (c as usize) * (h as usize) * (w as usize);
    let n_out = n_in * 4;
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
    let b_i = b as i32;
    let c_i = c as i32;
    let h_i = h as i32;
    let w_i = w as i32;

    macro_rules! go {
        ($t:ty, $func:expr) => {{
            let in_v = unsafe {
                input
                    .slice(input_off..input_off + n_in * esz)
                    .transmute::<$t>(n_in)
                    .ok_or_else(|| SynaptixError::Cuda("upsample2x: transmute input".into()))?
            };
            let mut out_s = output.slice_mut(output_off..output_off + n_out * esz);
            let mut out_v = unsafe {
                out_s
                    .transmute_mut::<$t>(n_out)
                    .ok_or_else(|| SynaptixError::Cuda("upsample2x: transmute output".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&in_v)
                .arg(&mut out_v)
                .arg(&b_i)
                .arg(&c_i)
                .arg(&h_i)
                .arg(&w_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch upsample2x: {e:?}")))?;
            }
        }};
    }

    match dtype {
        DType::F32 => go!(f32, &kernels.f32),
        DType::F16 => go!(f16, &kernels.f16),
        DType::BF16 => go!(bf16, &kernels.bf16),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "upsample2x: unsupported dtype {other:?}"
            )))
        }
    }
    Ok(())
}

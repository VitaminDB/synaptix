//! Parallel prefix sum (scan) per row, single-block. Каждый block обрабатывает
//! одну строку длины N ≤ 8192 (BLOCK=256 × MAX_PER_THREAD=32). Inclusive или
//! exclusive scan. F32/F16/BF16, fp32 accumulator.
//!
//! Алгоритм: chunked scan (Hillis-Steele на per-thread totals + fixup) —
//! O(N) work, O(log BLOCK + N/BLOCK) critical path. Для произвольно больших N
//! требуется multi-block scan c decoupled lookback — это будущая итерация.

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
const MAX_PER_THREAD: u32 = 32;
const MAX_N: u32 = BLOCK * MAX_PER_THREAD;

pub struct ParallelScanKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<ParallelScanKernels>)>>> = OnceLock::new();

impl ParallelScanKernels {
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
        let src = include_str!("../cu/scan/parallel_scan.cu");
        let module = compile_module(ctx, src, "parallel_scan.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "scan_sum_f32")?,
            f16: load_fn(&module, "scan_sum_f16")?,
            bf16: load_fn(&module, "scan_sum_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run<T: DeviceRepr>(
    kernels: &ParallelScanKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    y: &mut CudaSlice<T>,
    batch: u32,
    n: u32,
    inclusive: bool,
    dtype: DType,
) -> Result<()> {
    if n == 0 || n > MAX_N {
        return Err(SynaptixError::Cuda(format!(
            "parallel_scan: N={n} must be in [1, {MAX_N}]"
        )));
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        _ => {
            return Err(SynaptixError::Unsupported(
                "parallel_scan: dtype must be F32/F16/BF16",
            ))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let b_i = batch as i32;
    let n_i = n as i32;
    let inc_i = if inclusive { 1_i32 } else { 0_i32 };
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(&mut *y).arg(&b_i).arg(&n_i).arg(&inc_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch parallel_scan: {e:?}")))?;
    }
    Ok(())
}

pub fn run_f32(
    kernels: &ParallelScanKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    y: &mut CudaSlice<f32>,
    batch: u32,
    n: u32,
    inclusive: bool,
) -> Result<()> {
    run::<f32>(kernels, stream, x, y, batch, n, inclusive, DType::F32)
}

pub fn run_f16(
    kernels: &ParallelScanKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    y: &mut CudaSlice<f16>,
    batch: u32,
    n: u32,
    inclusive: bool,
) -> Result<()> {
    run::<f16>(kernels, stream, x, y, batch, n, inclusive, DType::F16)
}

pub fn run_bf16(
    kernels: &ParallelScanKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    y: &mut CudaSlice<bf16>,
    batch: u32,
    n: u32,
    inclusive: bool,
) -> Result<()> {
    run::<bf16>(kernels, stream, x, y, batch, n, inclusive, DType::BF16)
}

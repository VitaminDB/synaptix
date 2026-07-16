//! Fused cross-entropy: softmax + nll loss за один pass.
//!
//! `loss[b] = log_sum_exp(logits[b]) - logits[b, target[b]]`. Online stable
//! softmax (Milakov-Gimelshein) экономит один проход по logits vs.
//! двух-passового подхода. F32/F16/BF16 logits, i32 targets, F32 losses.

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

pub struct CrossEntropyKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<CrossEntropyKernels>)>>> = OnceLock::new();

impl CrossEntropyKernels {
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
        let src = include_str!("../cu/fused/loss/cross_entropy.cu");
        let module = compile_module(ctx, src, "cross_entropy.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "cross_entropy_f32")?,
            f16: load_fn(&module, "cross_entropy_f16")?,
            bf16: load_fn(&module, "cross_entropy_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run<T: DeviceRepr>(
    kernels: &CrossEntropyKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<T>,
    targets: &CudaSlice<i32>,
    losses: &mut CudaSlice<f32>,
    batch: u32,
    vocab: u32,
    ignore_index: i32,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "cross_entropy: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(func);
    bld.arg(logits)
        .arg(targets)
        .arg(&mut *losses)
        .arg(&batch)
        .arg(&vocab)
        .arg(&ignore_index);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch cross_entropy: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn run_f32(
    kernels: &CrossEntropyKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<f32>,
    targets: &CudaSlice<i32>,
    losses: &mut CudaSlice<f32>,
    batch: u32,
    vocab: u32,
    ignore_index: i32,
) -> Result<()> {
    run::<f32>(
        kernels,
        stream,
        logits,
        targets,
        losses,
        batch,
        vocab,
        ignore_index,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_f16(
    kernels: &CrossEntropyKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<f16>,
    targets: &CudaSlice<i32>,
    losses: &mut CudaSlice<f32>,
    batch: u32,
    vocab: u32,
    ignore_index: i32,
) -> Result<()> {
    run::<f16>(
        kernels,
        stream,
        logits,
        targets,
        losses,
        batch,
        vocab,
        ignore_index,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_bf16(
    kernels: &CrossEntropyKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<bf16>,
    targets: &CudaSlice<i32>,
    losses: &mut CudaSlice<f32>,
    batch: u32,
    vocab: u32,
    ignore_index: i32,
) -> Result<()> {
    run::<bf16>(
        kernels,
        stream,
        logits,
        targets,
        losses,
        batch,
        vocab,
        ignore_index,
        DType::BF16,
    )
}

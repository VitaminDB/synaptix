//! Top-K logits per row для sampling. F32/F16/BF16 inputs, F32 values + i32
//! indices outputs. Один CUDA block per row, K passes block-wide arg-max
//! reduction с bitmask в dynamic shared memory.
//!
//! Лимит V: smem_mask = ceil(V/32) * 4 bytes. SMEM_OPT_IN_BYTES = 99 KiB
//! даёт потолок V ≈ 810 000, чего хватает для всех практических LLM-vocab'ов.

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DeviceRepr, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const BLOCK: u32 = 256;
const SMEM_OPT_IN_BYTES: i32 = 96 * 1024;
const MAX_VOCAB: u32 = (SMEM_OPT_IN_BYTES as u32 / 4) * 32;

pub struct TopkKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<TopkKernels>)>>> = OnceLock::new();

impl TopkKernels {
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
        let src = include_str!("../cu/reduction/topk.cu");
        let module = compile_module_with_opts(ctx, src, "topk.cu", &[], Some("sm_120a"))?;
        let f32 = load_fn(&module, "topk_f32")?;
        let f16 = load_fn(&module, "topk_f16")?;
        let bf16 = load_fn(&module, "topk_bf16")?;
        for f in [&f32, &f16, &bf16] {
            f.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SMEM_OPT_IN_BYTES,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attribute topk shared: {e:?}")))?;
        }
        let new = Arc::new(Self {
            f32,
            f16,
            bf16,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn run<T: DeviceRepr>(
    kernels: &TopkKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<T>,
    out_vals: &mut CudaSlice<f32>,
    out_idx: &mut CudaSlice<i32>,
    batch: u32,
    vocab: u32,
    k: u32,
    dtype: DType,
) -> Result<()> {
    if vocab > MAX_VOCAB {
        return Err(SynaptixError::Cuda(format!(
            "topk: V={vocab} > MAX_VOCAB={MAX_VOCAB} (smem mask cap)"
        )));
    }
    if k == 0 || k > vocab {
        return Err(SynaptixError::Cuda(format!(
            "topk: K={k} must be 0 < K <= V={vocab}"
        )));
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        _ => {
            return Err(SynaptixError::Unsupported(
                "topk: dtype must be F32/F16/BF16",
            ))
        }
    };
    let mask_words = (vocab + 31) / 32;
    // smem = mask (u32 × mask_words) + s_block_v (f32 × BLOCK) + s_block_i (i32 × BLOCK)
    let smem_bytes = mask_words * 4 + BLOCK * 4 + BLOCK * 4;
    let cfg = LaunchConfig {
        grid_dim: (batch.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem_bytes,
    };
    let b_i = batch as i32;
    let v_i = vocab as i32;
    let k_i = k as i32;
    let mut bld = stream.launch_builder(func);
    bld.arg(logits)
        .arg(&mut *out_vals)
        .arg(&mut *out_idx)
        .arg(&b_i)
        .arg(&v_i)
        .arg(&k_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch topk: {e:?}")))?;
    }
    Ok(())
}

pub fn run_f32(
    kernels: &TopkKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<f32>,
    out_vals: &mut CudaSlice<f32>,
    out_idx: &mut CudaSlice<i32>,
    batch: u32,
    vocab: u32,
    k: u32,
) -> Result<()> {
    run::<f32>(
        kernels,
        stream,
        logits,
        out_vals,
        out_idx,
        batch,
        vocab,
        k,
        DType::F32,
    )
}

pub fn run_f16(
    kernels: &TopkKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<f16>,
    out_vals: &mut CudaSlice<f32>,
    out_idx: &mut CudaSlice<i32>,
    batch: u32,
    vocab: u32,
    k: u32,
) -> Result<()> {
    run::<f16>(
        kernels,
        stream,
        logits,
        out_vals,
        out_idx,
        batch,
        vocab,
        k,
        DType::F16,
    )
}

pub fn run_bf16(
    kernels: &TopkKernels,
    stream: &Arc<CudaStream>,
    logits: &CudaSlice<bf16>,
    out_vals: &mut CudaSlice<f32>,
    out_idx: &mut CudaSlice<i32>,
    batch: u32,
    vocab: u32,
    k: u32,
) -> Result<()> {
    run::<bf16>(
        kernels,
        stream,
        logits,
        out_vals,
        out_idx,
        batch,
        vocab,
        k,
        DType::BF16,
    )
}

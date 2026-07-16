//! MoE token routing (row-permutation copy):
//! - `scatter`: `out[i, :] = x[idx[i], :]` — раскладка токенов под экспертов;
//! - `gather`:  `out[idx[i], :] = x[i, :]` — возврат выходов на исходные позиции.
//!
//! `x`/`out` [N, D] row-major, `idx` [N] (u32). F32/F16/BF16. Семантика совпадает
//! с `synaptix_ops::ffn::moe::{scatter_tokens, gather_tokens}`. Для `gather` буфер
//! `out` должен быть предварительно занулён, если `idx` не полная перестановка.

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

pub struct MoeDispatchKernels {
    _module: Arc<CudaModule>,
    scatter_f32: CudaFunction,
    scatter_f16: CudaFunction,
    scatter_bf16: CudaFunction,
    gather_f32: CudaFunction,
    gather_f16: CudaFunction,
    gather_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<MoeDispatchKernels>)>>> = OnceLock::new();

impl MoeDispatchKernels {
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
        let src = include_str!("../cu/fused/moe/moe_dispatch.cu");
        let module = compile_module(ctx, src, "moe_dispatch.cu")?;
        let new = Arc::new(Self {
            scatter_f32: load_fn(&module, "moe_scatter_f32")?,
            scatter_f16: load_fn(&module, "moe_scatter_f16")?,
            scatter_bf16: load_fn(&module, "moe_scatter_bf16")?,
            gather_f32: load_fn(&module, "moe_gather_f32")?,
            gather_f16: load_fn(&module, "moe_gather_f16")?,
            gather_bf16: load_fn(&module, "moe_gather_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn launch<T: DeviceRepr>(
    func: &CudaFunction,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<T>,
    n: u32,
    d: u32,
    tag: &str,
) -> Result<()> {
    if n == 0 || d == 0 {
        return Ok(());
    }
    let total = (n as u64) * (d as u64);
    let cfg = LaunchConfig {
        grid_dim: (total.div_ceil(BLOCK as u64) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_i, d_i) = (n as i32, d as i32);
    let mut bld = stream.launch_builder(func);
    bld.arg(x).arg(idx).arg(&mut *out).arg(&n_i).arg(&d_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch {tag}: {e:?}")))?;
    }
    Ok(())
}

/// `out[i, :] = x[idx[i], :]` (сбор строк по индексам).
pub fn moe_scatter<T: DeviceRepr>(
    kernels: &MoeDispatchKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<T>,
    n: u32,
    d: u32,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.scatter_f32,
        DType::F16 => &kernels.scatter_f16,
        DType::BF16 => &kernels.scatter_bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "moe_scatter: unsupported dtype {other:?}"
            )))
        }
    };
    launch(func, stream, x, idx, out, n, d, "moe_scatter")
}

/// `out[idx[i], :] = x[i, :]` (раскладка строк по индексам, обратная scatter).
pub fn moe_gather<T: DeviceRepr>(
    kernels: &MoeDispatchKernels,
    stream: &Arc<CudaStream>,
    x: &CudaSlice<T>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<T>,
    n: u32,
    d: u32,
    dtype: DType,
) -> Result<()> {
    let func = match dtype {
        DType::F32 => &kernels.gather_f32,
        DType::F16 => &kernels.gather_f16,
        DType::BF16 => &kernels.gather_bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "moe_gather: unsupported dtype {other:?}"
            )))
        }
    };
    launch(func, stream, x, idx, out, n, d, "moe_gather")
}

pub fn moe_scatter_f32(
    k: &MoeDispatchKernels,
    s: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<f32>,
    n: u32,
    d: u32,
) -> Result<()> {
    moe_scatter::<f32>(k, s, x, idx, out, n, d, DType::F32)
}
pub fn moe_scatter_f16(
    k: &MoeDispatchKernels,
    s: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<f16>,
    n: u32,
    d: u32,
) -> Result<()> {
    moe_scatter::<f16>(k, s, x, idx, out, n, d, DType::F16)
}
pub fn moe_scatter_bf16(
    k: &MoeDispatchKernels,
    s: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<bf16>,
    n: u32,
    d: u32,
) -> Result<()> {
    moe_scatter::<bf16>(k, s, x, idx, out, n, d, DType::BF16)
}
pub fn moe_gather_f32(
    k: &MoeDispatchKernels,
    s: &Arc<CudaStream>,
    x: &CudaSlice<f32>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<f32>,
    n: u32,
    d: u32,
) -> Result<()> {
    moe_gather::<f32>(k, s, x, idx, out, n, d, DType::F32)
}
pub fn moe_gather_f16(
    k: &MoeDispatchKernels,
    s: &Arc<CudaStream>,
    x: &CudaSlice<f16>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<f16>,
    n: u32,
    d: u32,
) -> Result<()> {
    moe_gather::<f16>(k, s, x, idx, out, n, d, DType::F16)
}
pub fn moe_gather_bf16(
    k: &MoeDispatchKernels,
    s: &Arc<CudaStream>,
    x: &CudaSlice<bf16>,
    idx: &CudaSlice<u32>,
    out: &mut CudaSlice<bf16>,
    n: u32,
    d: u32,
) -> Result<()> {
    moe_gather::<bf16>(k, s, x, idx, out, n, d, DType::BF16)
}

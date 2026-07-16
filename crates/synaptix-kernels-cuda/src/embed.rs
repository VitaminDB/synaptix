//! Token-embedding gather: `out[t, :] = table[ids[t], :]`.
//!
//! `table` [V, D] row-major, `ids` [N] (u32), `out` [N, D] row-major.
//! Один thread = один выходной элемент. OOB id (>= V) → строка нулей.
//! F32/F16/BF16 (чистое копирование, lossless). Семантика совпадает с
//! `synaptix_ops::embed::token_embedding` (index_select по dim 0).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, DeviceRepr,
    LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 256;

pub struct EmbedKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<EmbedKernels>)>>> = OnceLock::new();

impl EmbedKernels {
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
        let src = include_str!("cu/embed/embed.cu");
        let module = compile_module(ctx, src, "embed.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "embed_gather_f32")?,
            f16: load_fn(&module, "embed_gather_f16")?,
            bf16: load_fn(&module, "embed_gather_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `out[t, :] = table[ids[t], :]`. `table` [vocab, dim], `ids` [n_ids] (u32),
/// `out` [n_ids, dim]. OOB id → строка нулей.
#[allow(clippy::too_many_arguments)]
pub fn embed_gather<T: DeviceRepr>(
    kernels: &EmbedKernels,
    stream: &Arc<CudaStream>,
    table: &CudaSlice<T>,
    ids: &CudaSlice<u32>,
    out: &mut CudaSlice<T>,
    n_ids: u32,
    dim: u32,
    vocab: u32,
    dtype: DType,
) -> Result<()> {
    if n_ids == 0 || dim == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "embed: unsupported dtype {other:?}"
            )))
        }
    };
    let total = (n_ids as u64) * (dim as u64);
    let grid = total.div_ceil(BLOCK as u64) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_ids_i, dim_i, vocab_i) = (n_ids as i32, dim as i32, vocab as i32);
    let mut bld = stream.launch_builder(func);
    bld.arg(table)
        .arg(ids)
        .arg(&mut *out)
        .arg(&n_ids_i)
        .arg(&dim_i)
        .arg(&vocab_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch embed_gather: {e:?}")))?;
    }
    Ok(())
}

pub fn embed_gather_f32(
    kernels: &EmbedKernels,
    stream: &Arc<CudaStream>,
    table: &CudaSlice<f32>,
    ids: &CudaSlice<u32>,
    out: &mut CudaSlice<f32>,
    n_ids: u32,
    dim: u32,
    vocab: u32,
) -> Result<()> {
    embed_gather::<f32>(
        kernels,
        stream,
        table,
        ids,
        out,
        n_ids,
        dim,
        vocab,
        DType::F32,
    )
}

pub fn embed_gather_f16(
    kernels: &EmbedKernels,
    stream: &Arc<CudaStream>,
    table: &CudaSlice<f16>,
    ids: &CudaSlice<u32>,
    out: &mut CudaSlice<f16>,
    n_ids: u32,
    dim: u32,
    vocab: u32,
) -> Result<()> {
    embed_gather::<f16>(
        kernels,
        stream,
        table,
        ids,
        out,
        n_ids,
        dim,
        vocab,
        DType::F16,
    )
}

pub fn embed_gather_bf16(
    kernels: &EmbedKernels,
    stream: &Arc<CudaStream>,
    table: &CudaSlice<bf16>,
    ids: &CudaSlice<u32>,
    out: &mut CudaSlice<bf16>,
    n_ids: u32,
    dim: u32,
    vocab: u32,
) -> Result<()> {
    embed_gather::<bf16>(
        kernels,
        stream,
        table,
        ids,
        out,
        n_ids,
        dim,
        vocab,
        DType::BF16,
    )
}

/// Embed-gather из untyped `u8`-storage с device-резидентными `ids` (для
/// `Backend::embed_gather` / CUDA-graph decode). Читает индексы из device-памяти
/// (`ids_dev: &CudaView<u32>`) — без host round-trip (в отличие от `index_select`,
/// который `clone_dtoh`'ит индексы и ломает capture). `table` `[vocab, dim]`,
/// `out` `[n_ids, dim]`, оба в `dtype`. Принимает byte-offset'ы.
#[allow(clippy::too_many_arguments)]
pub fn embed_gather_u8(
    kernels: &EmbedKernels,
    stream: &Arc<CudaStream>,
    table: &CudaSlice<u8>,
    table_off: usize,
    ids_dev: &CudaView<u32>,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    n_ids: u32,
    dim: u32,
    vocab: u32,
    dtype: DType,
) -> Result<()> {
    if n_ids == 0 || dim == 0 {
        return Ok(());
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "embed u8: unsupported dtype {other:?}"
            )))
        }
    };
    let esz = (dtype.size_in_bits() / 8) as usize;
    let table_n = (vocab as usize) * (dim as usize);
    let out_n = (n_ids as usize) * (dim as usize);
    let total = (n_ids as u64) * (dim as u64);
    let grid = total.div_ceil(BLOCK as u64) as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let (n_ids_i, dim_i, vocab_i) = (n_ids as i32, dim as i32, vocab as i32);
    macro_rules! go {
        ($t:ty) => {{
            let table_v = unsafe {
                table
                    .slice(table_off..table_off + table_n * esz)
                    .transmute::<$t>(table_n)
                    .ok_or_else(|| SynaptixError::Cuda("embed u8: transmute table".into()))?
            };
            let mut out_s = out.slice_mut(out_off..out_off + out_n * esz);
            let mut out_v = unsafe {
                out_s
                    .transmute_mut::<$t>(out_n)
                    .ok_or_else(|| SynaptixError::Cuda("embed u8: transmute out".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&table_v)
                .arg(ids_dev)
                .arg(&mut out_v)
                .arg(&n_ids_i)
                .arg(&dim_i)
                .arg(&vocab_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch embed_gather_u8: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F32 => go!(f32),
        DType::F16 => go!(f16),
        DType::BF16 => go!(bf16),
        _ => unreachable!(),
    }
    Ok(())
}

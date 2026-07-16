//! KV-cache append slice — F16 / BF16 (через u16 bitcast) / F32.
//!
//! Scatter-write нового `(B, kv, T_new, hd)` тензора в preallocated
//! `(B, kv, max_seq_len, hd)` ring-buffer на позицию `seq_pos`. Заменяет
//! `Tensor::cat` (выделение нового тензора + копия prev+new) — основной
//! win для decode-step pipeline.
//!
//! Два варианта: immediate `seq_pos: u32` (для prefill) и device-resident
//! `seq_pos_dev: &CudaSlice<u32>` (для CUDA graph capture / replay).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, LaunchConfig,
    PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 128;

pub struct KvAppendKernels {
    _module: Arc<CudaModule>,
    f16: CudaFunction,
    bf16: CudaFunction,
    f32: CudaFunction,
    f16_dev: CudaFunction,
    bf16_dev: CudaFunction,
    f32_dev: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<KvAppendKernels>)>>> = OnceLock::new();

impl KvAppendKernels {
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
        let src = include_str!("../cu/elementwise/kv_append.cu");
        let module = compile_module(ctx, src, "kv_append.cu")?;
        let new = Arc::new(Self {
            f16: load_fn(&module, "kv_append_slice_f16")?,
            bf16: load_fn(&module, "kv_append_slice_bf16")?,
            f32: load_fn(&module, "kv_append_slice_f32")?,
            f16_dev: load_fn(&module, "kv_append_slice_f16_dev")?,
            bf16_dev: load_fn(&module, "kv_append_slice_bf16_dev")?,
            f32_dev: load_fn(&module, "kv_append_slice_f32_dev")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn n_h2_grid(b: u32, kv: u32, t_new: u32, hd: u32) -> u32 {
    let n_h2 = (b * kv * t_new * hd) / 2;
    n_h2.div_ceil(BLOCK)
}

fn n_grid(b: u32, kv: u32, t_new: u32, hd: u32) -> u32 {
    let n = b * kv * t_new * hd;
    n.div_ceil(BLOCK)
}

/// Append из untyped `u8`-storage (для `Backend::kv_append`). `src`/`dst` —
/// contiguous row-major: src `[B,kv,T_new,hd]`, dst `[B,kv,max_seq_len,hd]`.
/// Принимает byte-offset'ы (view с offset≠0). Immediate `seq_pos` (prefill+decode).
#[allow(clippy::too_many_arguments)]
pub fn append_u8(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos: u32,
    dtype: DType,
) -> Result<()> {
    let esz = (dtype.size_in_bits() / 8) as usize;
    let src_n = (b as usize) * (kv as usize) * (t_new as usize) * (hd as usize);
    let dst_n = (b as usize) * (kv as usize) * (max_seq_len as usize) * (hd as usize);
    let is_h2 = matches!(dtype, DType::F16 | DType::BF16);
    if is_h2 && hd % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "kv_append u8: hd={hd} must be even"
        )));
    }
    let grid = if is_h2 {
        n_h2_grid(b, kv, t_new, hd)
    } else {
        n_grid(b, kv, t_new, hd)
    }
    .max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! run {
        ($t:ty, $func:expr) => {{
            let src_v = unsafe {
                src.slice(src_off..src_off + src_n * esz)
                    .transmute::<$t>(src_n)
                    .ok_or_else(|| SynaptixError::Cuda("kv_append u8: transmute src".into()))?
            };
            let mut dst_s = dst.slice_mut(dst_off..dst_off + dst_n * esz);
            let mut dst_v = unsafe {
                dst_s
                    .transmute_mut::<$t>(dst_n)
                    .ok_or_else(|| SynaptixError::Cuda("kv_append u8: transmute dst".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&src_v)
                .arg(&mut dst_v)
                .arg(&b)
                .arg(&kv)
                .arg(&t_new)
                .arg(&hd)
                .arg(&max_seq_len)
                .arg(&seq_pos);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_u8: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => run!(f16, &kernels.f16),
        DType::BF16 => run!(bf16, &kernels.bf16),
        DType::F32 => run!(f32, &kernels.f32),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "kv_append u8: unsupported dtype {other:?}"
            )))
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn append_f16(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f16>,
    dst: &mut CudaSlice<f16>,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos: u32,
) -> Result<()> {
    if hd % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "kv_append f16: hd={hd} must be even (half2 vectorization)"
        )));
    }
    let cfg = LaunchConfig {
        grid_dim: (n_h2_grid(b, kv, t_new, hd).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.f16);
    bld.arg(src)
        .arg(&mut *dst)
        .arg(&b)
        .arg(&kv)
        .arg(&t_new)
        .arg(&hd)
        .arg(&max_seq_len)
        .arg(&seq_pos);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_f16: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn append_bf16(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos: u32,
) -> Result<()> {
    if hd % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "kv_append bf16: hd={hd} must be even"
        )));
    }
    let cfg = LaunchConfig {
        grid_dim: (n_h2_grid(b, kv, t_new, hd).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.bf16);
    bld.arg(src)
        .arg(&mut *dst)
        .arg(&b)
        .arg(&kv)
        .arg(&t_new)
        .arg(&hd)
        .arg(&max_seq_len)
        .arg(&seq_pos);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_bf16: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn append_f32(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos: u32,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (n_grid(b, kv, t_new, hd).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.f32);
    bld.arg(src)
        .arg(&mut *dst)
        .arg(&b)
        .arg(&kv)
        .arg(&t_new)
        .arg(&hd)
        .arg(&max_seq_len)
        .arg(&seq_pos);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_f32: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn append_f16_dev(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f16>,
    dst: &mut CudaSlice<f16>,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos_dev: &CudaSlice<u32>,
) -> Result<()> {
    if hd % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "kv_append f16_dev: hd={hd} must be even"
        )));
    }
    let cfg = LaunchConfig {
        grid_dim: (n_h2_grid(b, kv, t_new, hd).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.f16_dev);
    bld.arg(src)
        .arg(&mut *dst)
        .arg(&b)
        .arg(&kv)
        .arg(&t_new)
        .arg(&hd)
        .arg(&max_seq_len)
        .arg(seq_pos_dev);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_f16_dev: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn append_bf16_dev(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<bf16>,
    dst: &mut CudaSlice<bf16>,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos_dev: &CudaSlice<u32>,
) -> Result<()> {
    if hd % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "kv_append bf16_dev: hd={hd} must be even"
        )));
    }
    let cfg = LaunchConfig {
        grid_dim: (n_h2_grid(b, kv, t_new, hd).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.bf16_dev);
    bld.arg(src)
        .arg(&mut *dst)
        .arg(&b)
        .arg(&kv)
        .arg(&t_new)
        .arg(&hd)
        .arg(&max_seq_len)
        .arg(seq_pos_dev);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_bf16_dev: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn append_f32_dev(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<f32>,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos_dev: &CudaSlice<u32>,
) -> Result<()> {
    let cfg = LaunchConfig {
        grid_dim: (n_grid(b, kv, t_new, hd).max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(&kernels.f32_dev);
    bld.arg(src)
        .arg(&mut *dst)
        .arg(&b)
        .arg(&kv)
        .arg(&t_new)
        .arg(&hd)
        .arg(&max_seq_len)
        .arg(seq_pos_dev);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_f32_dev: {e:?}")))?;
    }
    Ok(())
}

/// Device-resident-position append из untyped `u8`-storage (для
/// `Backend::kv_append_dev` / CUDA-graph). Как [`append_u8`], но позиция слота
/// `seq_pos` приходит device-резидентным указателем `seq_pos_dev`
/// (`&CudaView<u32>`, 1 элемент) — launch config от значения не зависит, один
/// граф валиден для всех decode-позиций.
#[allow(clippy::too_many_arguments)]
pub fn append_u8_dev(
    kernels: &KvAppendKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    b: u32,
    kv: u32,
    t_new: u32,
    hd: u32,
    max_seq_len: u32,
    seq_pos_dev: &CudaView<u32>,
    dtype: DType,
) -> Result<()> {
    let esz = (dtype.size_in_bits() / 8) as usize;
    let src_n = (b as usize) * (kv as usize) * (t_new as usize) * (hd as usize);
    let dst_n = (b as usize) * (kv as usize) * (max_seq_len as usize) * (hd as usize);
    let is_h2 = matches!(dtype, DType::F16 | DType::BF16);
    if is_h2 && hd % 2 != 0 {
        return Err(SynaptixError::Cuda(format!(
            "kv_append u8_dev: hd={hd} must be even"
        )));
    }
    let grid = if is_h2 {
        n_h2_grid(b, kv, t_new, hd)
    } else {
        n_grid(b, kv, t_new, hd)
    }
    .max(1);
    let cfg = LaunchConfig {
        grid_dim: (grid, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    macro_rules! run {
        ($t:ty, $func:expr) => {{
            let src_v = unsafe {
                src.slice(src_off..src_off + src_n * esz)
                    .transmute::<$t>(src_n)
                    .ok_or_else(|| SynaptixError::Cuda("kv_append u8_dev: transmute src".into()))?
            };
            let mut dst_s = dst.slice_mut(dst_off..dst_off + dst_n * esz);
            let mut dst_v = unsafe {
                dst_s
                    .transmute_mut::<$t>(dst_n)
                    .ok_or_else(|| SynaptixError::Cuda("kv_append u8_dev: transmute dst".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&src_v)
                .arg(&mut dst_v)
                .arg(&b)
                .arg(&kv)
                .arg(&t_new)
                .arg(&hd)
                .arg(&max_seq_len)
                .arg(seq_pos_dev);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch kv_append_u8_dev: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => run!(f16, &kernels.f16_dev),
        DType::BF16 => run!(bf16, &kernels.bf16_dev),
        DType::F32 => run!(f32, &kernels.f32_dev),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "kv_append u8_dev: unsupported dtype {other:?}"
            )))
        }
    }
    Ok(())
}

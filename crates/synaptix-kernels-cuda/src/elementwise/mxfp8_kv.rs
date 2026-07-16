//! MXFP8-KV (Blackwell block-scale) quantizing append — BF16/F16 → MXFP8 E4M3 +
//! per-32-block E8M0 scale (U8).
//!
//! Scatter-write нового `(B,nkv,T_new,hd)` BF16/F16-тензора в preallocated MXFP8
//! `(B,nkv,max_seq,hd)` ring-buffer на позицию `seq_pos`, с записью per-32-block
//! E8M0-scale в `(B,nkv,max_seq,hd/32)` U8. Параллельно FP8 E4M3 (`fp8_kv.rs`).
//! Один CUDA-блок = одна (b,kv,token)-строка (per-32-block amax → E8M0 → encode).
//! F16-src нужен при квант-весах (compute=F16 → K/V проекции F16).

use std::sync::{Arc, OnceLock};

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, CudaView, LaunchConfig,
    PushKernelArg,
};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BLOCK: u32 = 128;

pub struct MxFp8KvKernels {
    _module: Arc<CudaModule>,
    append_bf16: CudaFunction,
    append_f16: CudaFunction,
    append_bf16_dev: CudaFunction,
    append_f16_dev: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<MxFp8KvKernels>)>>> = OnceLock::new();

impl MxFp8KvKernels {
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
        let src = include_str!("../cu/elementwise/mxfp8_kv.cu");
        let module = compile_module(ctx, src, "mxfp8_kv.cu")?;
        let new = Arc::new(Self {
            append_bf16: load_fn(&module, "kv_quant_append_mxfp8_bf16")?,
            append_f16: load_fn(&module, "kv_quant_append_mxfp8_f16")?,
            append_bf16_dev: load_fn(&module, "kv_quant_append_mxfp8_bf16_dev")?,
            append_f16_dev: load_fn(&module, "kv_quant_append_mxfp8_f16_dev")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// Квантизующий append BF16/F16 `src` `[B,nkv,T_new,hd]` → MXFP8 `dst`
/// `[B,nkv,max_seq,hd]` (u8 E4M3) + U8 `scale_dst` `[B,nkv,max_seq,hd/32]` (E8M0),
/// slot `seq_pos`. `src_dtype` ∈ {BF16, F16}. Все слайсы — untyped `u8` storage с
/// byte-offset'ами. Требует `hd % 32 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn quant_append_mxfp8_u8(
    kernels: &MxFp8KvKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    scale_dst: &mut CudaSlice<u8>,
    scale_off: usize,
    b: u32,
    nkv: u32,
    t_new: u32,
    hd: u32,
    max_seq: u32,
    seq_pos: u32,
    src_dtype: DType,
) -> Result<()> {
    if b == 0 || nkv == 0 || t_new == 0 || hd == 0 {
        return Ok(());
    }
    if hd % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("mxfp8 kv: hd={hd} must be %32")));
    }
    let nb = (hd / 32) as usize;
    let src_n = (b as usize) * (nkv as usize) * (t_new as usize) * (hd as usize);
    let dst_n = (b as usize) * (nkv as usize) * (max_seq as usize) * (hd as usize);
    let scale_n = (b as usize) * (nkv as usize) * (max_seq as usize) * nb;
    let rows = b * nkv * t_new;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut dst_v = dst.slice_mut(dst_off..dst_off + dst_n);
    let mut sc_v = scale_dst.slice_mut(scale_off..scale_off + scale_n);
    macro_rules! launch {
        ($t:ty, $func:expr) => {{
            let src_v = unsafe {
                src.slice(src_off..src_off + src_n * 2)
                    .transmute::<$t>(src_n)
                    .ok_or_else(|| SynaptixError::Cuda("mxfp8 kv: transmute src".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&src_v)
                .arg(&mut dst_v)
                .arg(&mut sc_v)
                .arg(&b)
                .arg(&nkv)
                .arg(&t_new)
                .arg(&hd)
                .arg(&max_seq)
                .arg(&seq_pos);
            unsafe {
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch kv_quant_append_mxfp8: {e:?}"))
                })?;
            }
        }};
    }
    match src_dtype {
        DType::BF16 => launch!(half::bf16, &kernels.append_bf16),
        DType::F16 => launch!(half::f16, &kernels.append_f16),
        _ => return Err(SynaptixError::Unsupported("mxfp8 kv: src dtype (BF16/F16)")),
    }
    Ok(())
}

/// Device-pos вариант [`quant_append_mxfp8_u8`] (CUDA-graph): `seq_pos` приходит
/// device-резидентным `&CudaView<u32>` (1 элемент) — один граф валиден для всех
/// decode-позиций.
#[allow(clippy::too_many_arguments)]
pub fn quant_append_mxfp8_u8_dev(
    kernels: &MxFp8KvKernels,
    stream: &Arc<CudaStream>,
    src: &CudaSlice<u8>,
    src_off: usize,
    dst: &mut CudaSlice<u8>,
    dst_off: usize,
    scale_dst: &mut CudaSlice<u8>,
    scale_off: usize,
    b: u32,
    nkv: u32,
    t_new: u32,
    hd: u32,
    max_seq: u32,
    seq_pos_dev: &CudaView<u32>,
    src_dtype: DType,
) -> Result<()> {
    if b == 0 || nkv == 0 || t_new == 0 || hd == 0 {
        return Ok(());
    }
    if hd % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("mxfp8 kv dev: hd={hd} must be %32")));
    }
    let nb = (hd / 32) as usize;
    let src_n = (b as usize) * (nkv as usize) * (t_new as usize) * (hd as usize);
    let dst_n = (b as usize) * (nkv as usize) * (max_seq as usize) * (hd as usize);
    let scale_n = (b as usize) * (nkv as usize) * (max_seq as usize) * nb;
    let rows = b * nkv * t_new;
    let cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut dst_v = dst.slice_mut(dst_off..dst_off + dst_n);
    let mut sc_v = scale_dst.slice_mut(scale_off..scale_off + scale_n);
    macro_rules! launch {
        ($t:ty, $func:expr) => {{
            let src_v = unsafe {
                src.slice(src_off..src_off + src_n * 2)
                    .transmute::<$t>(src_n)
                    .ok_or_else(|| SynaptixError::Cuda("mxfp8 kv dev: transmute src".into()))?
            };
            let mut bld = stream.launch_builder($func);
            bld.arg(&src_v)
                .arg(&mut dst_v)
                .arg(&mut sc_v)
                .arg(&b)
                .arg(&nkv)
                .arg(&t_new)
                .arg(&hd)
                .arg(&max_seq)
                .arg(seq_pos_dev);
            unsafe {
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch kv_quant_append_mxfp8_dev: {e:?}"))
                })?;
            }
        }};
    }
    match src_dtype {
        DType::BF16 => launch!(half::bf16, &kernels.append_bf16_dev),
        DType::F16 => launch!(half::f16, &kernels.append_f16_dev),
        _ => return Err(SynaptixError::Unsupported("mxfp8 kv dev: src dtype (BF16/F16)")),
    }
    Ok(())
}

//! Flash-decoding split-K (decode path, T_q обычно = 1).
//!
//! KV-измерение разбивается на `split_k` сегментов (grid.y); каждый блок считает
//! online-softmax по своему сегменту и пишет НЕнормализованный partial
//! `(m, l, acc[D])` в F32. Merge-ядро объединяет partial'ы через
//! online-softmax-merge и нормализует выход. Устраняет underoccupancy на decode
//! (T_q=1), где `grid = B·NH` слишком мал относительно числа SM.
//!
//! Layout совпадает с `sdpa_f32_acc`/`flash_v2`: q (B, NH, Tq, D),
//! k/v (B, NKV, Tkv, D), out (B, NH, Tq, D) row-major; GQA h_kv = h/(NH/NKV);
//! causal q_pos = (Tkv≥Tq) ? Tkv−Tq+ti : ti. F32/F16/BF16, аккумулятор F32.

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

const BLOCK: u32 = 128;
const SPLIT_K_MAX: u32 = 32;

pub struct FlashDecodeKernels {
    _module: Arc<CudaModule>,
    split_f32: CudaFunction,
    split_f16: CudaFunction,
    split_bf16: CudaFunction,
    split_mxfp8_f32: CudaFunction,
    split_mxfp8_f16: CudaFunction,
    split_mxfp8_bf16: CudaFunction,
    split_mxfp8_f32_dev: CudaFunction,
    split_mxfp8_f16_dev: CudaFunction,
    split_mxfp8_bf16_dev: CudaFunction,
    split_f32_dev: CudaFunction,
    split_f16_dev: CudaFunction,
    split_bf16_dev: CudaFunction,
    merge_f32: CudaFunction,
    merge_f16: CudaFunction,
    merge_bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashDecodeKernels>)>>> = OnceLock::new();

impl FlashDecodeKernels {
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
        let src = include_str!("../cu/fused/attention/flash_decode.cu");
        let module = compile_module(ctx, src, "flash_decode.cu")?;
        let new = Arc::new(Self {
            split_f32: load_fn(&module, "flash_decode_split_f32")?,
            split_f16: load_fn(&module, "flash_decode_split_f16")?,
            split_bf16: load_fn(&module, "flash_decode_split_bf16")?,
            split_mxfp8_f32: load_fn(&module, "flash_decode_split_mxfp8_f32")?,
            split_mxfp8_f16: load_fn(&module, "flash_decode_split_mxfp8_f16")?,
            split_mxfp8_bf16: load_fn(&module, "flash_decode_split_mxfp8_bf16")?,
            split_mxfp8_f32_dev: load_fn(&module, "flash_decode_split_mxfp8_f32_dev")?,
            split_mxfp8_f16_dev: load_fn(&module, "flash_decode_split_mxfp8_f16_dev")?,
            split_mxfp8_bf16_dev: load_fn(&module, "flash_decode_split_mxfp8_bf16_dev")?,
            split_f32_dev: load_fn(&module, "flash_decode_split_f32_dev")?,
            split_f16_dev: load_fn(&module, "flash_decode_split_f16_dev")?,
            split_bf16_dev: load_fn(&module, "flash_decode_split_bf16_dev")?,
            merge_f32: load_fn(&module, "flash_decode_merge_f32")?,
            merge_f16: load_fn(&module, "flash_decode_merge_f16")?,
            merge_bf16: load_fn(&module, "flash_decode_merge_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn validate(nh: u32, nkv: u32, d: u32, split_k: u32) -> Result<()> {
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_decode: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    if !(1..=SPLIT_K_MAX).contains(&split_k) {
        return Err(SynaptixError::Cuda(format!(
            "flash_decode: split_k must be in [1,{SPLIT_K_MAX}], got {split_k}"
        )));
    }
    // dynamic smem split = (2*D + TILE_KV) * 4 байта; держим D в разумных рамках.
    if d == 0 || d > 1024 {
        return Err(SynaptixError::Cuda(format!(
            "flash_decode: D={d} вне поддерживаемого диапазона [1,1024]"
        )));
    }
    Ok(())
}

/// `out = softmax(scale·Q·Kᵀ + causal_mask)·V` через split-K decode + merge.
/// Аллоцирует F32-partial-буферы на `stream`, запускает split- и merge-ядра.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode<T: DeviceRepr>(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<T>,
    k: &CudaSlice<T>,
    v: &CudaSlice<T>,
    out: &mut CudaSlice<T>,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
    t_stride: u32,
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 || d == 0 {
        return Ok(());
    }
    validate(nh, nkv, d, split_k)?;
    let (split_fn, merge_fn) = match dtype {
        DType::F32 => (&kernels.split_f32, &kernels.merge_f32),
        DType::F16 => (&kernels.split_f16, &kernels.merge_f16),
        DType::BF16 => (&kernels.split_bf16, &kernels.merge_bf16),
        other => {
            return Err(SynaptixError::Cuda(format!(
                "flash_decode: unsupported dtype {other:?}"
            )))
        }
    };

    let rows = b * nh * t_q;
    let n_partials = (rows as usize) * (split_k as usize);
    let mut partial_acc = stream
        .alloc_zeros::<f32>(n_partials * d as usize)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_acc: {e:?}")))?;
    let mut partial_m = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_m: {e:?}")))?;
    let mut partial_l = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_l: {e:?}")))?;

    let (b_i, nh_i, nkv_i, tq_i, tkv_i, d_i, sk_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        t_q as i32,
        t_kv as i32,
        d as i32,
        split_k as i32,
    );
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;

    // ── Split ──
    let smem = (2 * d + 128) * 4;
    let split_cfg = LaunchConfig {
        grid_dim: (rows, split_k, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let mut bld = stream.launch_builder(split_fn);
    bld.arg(q)
        .arg(k)
        .arg(v)
        .arg(&mut partial_acc)
        .arg(&mut partial_m)
        .arg(&mut partial_l)
        .arg(&b_i)
        .arg(&nh_i)
        .arg(&nkv_i)
        .arg(&tq_i)
        .arg(&tkv_i)
        .arg(&d_i)
        .arg(&scale)
        .arg(&causal_i)
        .arg(&sk_i)
        .arg(&ts_i);
    unsafe {
        bld.launch(split_cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch flash_decode_split: {e:?}")))?;
    }

    // ── Merge ──
    let merge_cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut bld = stream.launch_builder(merge_fn);
    bld.arg(&partial_acc)
        .arg(&partial_m)
        .arg(&partial_l)
        .arg(&mut *out)
        .arg(&b_i)
        .arg(&nh_i)
        .arg(&tq_i)
        .arg(&d_i)
        .arg(&sk_i);
    unsafe {
        bld.launch(merge_cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch flash_decode_merge: {e:?}")))?;
    }
    Ok(())
}

/// Flash-decode из untyped `u8`-storage (для `Backend::flash_attention`).
/// `q` [B,NH,Tq,D], `k`/`v` [B,NKV,Tkv,D], `out` [B,NH,Tq,D] — все `dtype`.
/// GQA обрабатывается ядром (nh/nkv), causal/scale, F32-аккумулятор. Принимает
/// byte-offset'ы (q/k/v/out могут быть view с offset≠0).
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_u8(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
    t_stride: u32,
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 || d == 0 {
        return Ok(());
    }
    validate(nh, nkv, d, split_k)?;
    let esz = (dtype.size_in_bits() / 8) as usize;
    // Физический row-stride dim-T: при strided KV-буфере t_stride>0 покрывает весь
    // preallocated регион (b*nkv*t_stride*hd), иначе contiguous = t_kv.
    let t_stride_eff = if t_stride > 0 { t_stride } else { t_kv } as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let rows = b * nh * t_q;
    let n_partials = (rows as usize) * (split_k as usize);
    let mut partial_acc = stream
        .alloc_zeros::<f32>(n_partials * d as usize)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_acc: {e:?}")))?;
    let mut partial_m = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_m: {e:?}")))?;
    let mut partial_l = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_l: {e:?}")))?;
    let (b_i, nh_i, nkv_i, tq_i, tkv_i, d_i, sk_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        t_q as i32,
        t_kv as i32,
        d as i32,
        split_k as i32,
    );
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    let smem = (2 * d + 128) * 4;
    let split_cfg = LaunchConfig {
        grid_dim: (rows, split_k, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let merge_cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    macro_rules! run {
        ($t:ty, $split:expr, $merge:expr) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash: transmute q".into()))?
            };
            let k_v = unsafe {
                k.slice(k_off..k_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash: transmute k".into()))?
            };
            let v_v = unsafe {
                v.slice(v_off..v_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash: transmute v".into()))?
            };
            {
                let mut bld = stream.launch_builder($split);
                bld.arg(&q_v)
                    .arg(&k_v)
                    .arg(&v_v)
                    .arg(&mut partial_acc)
                    .arg(&mut partial_m)
                    .arg(&mut partial_l)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&tq_i)
                    .arg(&tkv_i)
                    .arg(&d_i)
                    .arg(&scale)
                    .arg(&causal_i)
                    .arg(&sk_i)
                    .arg(&ts_i);
                unsafe {
                    bld.launch(split_cfg)
                        .map_err(|e| SynaptixError::Cuda(format!("launch flash split: {e:?}")))?;
                }
            }
            {
                let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
                let mut o_v = unsafe {
                    o_s.transmute_mut::<$t>(q_n)
                        .ok_or_else(|| SynaptixError::Cuda("flash: transmute out".into()))?
                };
                let mut bld = stream.launch_builder($merge);
                bld.arg(&partial_acc)
                    .arg(&partial_m)
                    .arg(&partial_l)
                    .arg(&mut o_v)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&tq_i)
                    .arg(&d_i)
                    .arg(&sk_i);
                unsafe {
                    bld.launch(merge_cfg)
                        .map_err(|e| SynaptixError::Cuda(format!("launch flash merge: {e:?}")))?;
                }
            }
        }};
    }

    match dtype {
        DType::F32 => run!(f32, &kernels.split_f32, &kernels.merge_f32),
        DType::F16 => run!(f16, &kernels.split_f16, &kernels.merge_f16),
        DType::BF16 => run!(bf16, &kernels.split_bf16, &kernels.merge_bf16),
        _ => return Err(SynaptixError::Unsupported("flash_decode_u8: dtype")),
    }
    Ok(())
}

/// Device-resident-length flash-decode (для CUDA-graph capture/replay). Как
/// [`flash_decode_u8`], но активная длина KV `t_kv` приходит device-резидентным
/// указателем `tkv_dev` (`&CudaSlice<u32>`, 1 элемент) — launch config от
/// значения не зависит, поэтому один граф валиден для всех decode-позиций
/// (значение обновляется `memcpy_htod` перед каждым replay'ем). Требует
/// preallocated KV-буфер (`t_stride > 0`): слайс размечается по физическому
/// `t_stride`, ядро читает только `[0, *tkv_dev)`. Tq обычно = 1.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_u8_dev(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    tkv_dev: &CudaView<u32>,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
    t_stride: u32,
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || d == 0 {
        return Ok(());
    }
    if t_stride == 0 {
        return Err(SynaptixError::Cuda(
            "flash_decode_u8_dev: требуется preallocated буфер (t_stride > 0)".into(),
        ));
    }
    validate(nh, nkv, d, split_k)?;
    let esz = (dtype.size_in_bits() / 8) as usize;
    let t_stride_eff = t_stride as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let rows = b * nh * t_q;
    let n_partials = (rows as usize) * (split_k as usize);
    let mut partial_acc = stream
        .alloc_zeros::<f32>(n_partials * d as usize)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_acc: {e:?}")))?;
    let mut partial_m = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_m: {e:?}")))?;
    let mut partial_l = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_l: {e:?}")))?;
    let (b_i, nh_i, nkv_i, tq_i, d_i, sk_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        t_q as i32,
        d as i32,
        split_k as i32,
    );
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    let smem = (2 * d + 128) * 4;
    let split_cfg = LaunchConfig {
        grid_dim: (rows, split_k, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let merge_cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    macro_rules! run {
        ($t:ty, $split:expr, $merge:expr) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash dev: transmute q".into()))?
            };
            let k_v = unsafe {
                k.slice(k_off..k_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash dev: transmute k".into()))?
            };
            let v_v = unsafe {
                v.slice(v_off..v_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash dev: transmute v".into()))?
            };
            {
                let mut bld = stream.launch_builder($split);
                bld.arg(&q_v)
                    .arg(&k_v)
                    .arg(&v_v)
                    .arg(&mut partial_acc)
                    .arg(&mut partial_m)
                    .arg(&mut partial_l)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&tq_i)
                    .arg(tkv_dev)
                    .arg(&d_i)
                    .arg(&scale)
                    .arg(&causal_i)
                    .arg(&sk_i)
                    .arg(&ts_i);
                unsafe {
                    bld.launch(split_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash dev split: {e:?}"))
                    })?;
                }
            }
            {
                let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
                let mut o_v = unsafe {
                    o_s.transmute_mut::<$t>(q_n)
                        .ok_or_else(|| SynaptixError::Cuda("flash dev: transmute out".into()))?
                };
                let mut bld = stream.launch_builder($merge);
                bld.arg(&partial_acc)
                    .arg(&partial_m)
                    .arg(&partial_l)
                    .arg(&mut o_v)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&tq_i)
                    .arg(&d_i)
                    .arg(&sk_i);
                unsafe {
                    bld.launch(merge_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash dev merge: {e:?}"))
                    })?;
                }
            }
        }};
    }

    match dtype {
        DType::F32 => run!(f32, &kernels.split_f32_dev, &kernels.merge_f32),
        DType::F16 => run!(f16, &kernels.split_f16_dev, &kernels.merge_f16),
        DType::BF16 => run!(bf16, &kernels.split_bf16_dev, &kernels.merge_bf16),
        _ => return Err(SynaptixError::Unsupported("flash_decode_u8_dev: dtype")),
    }
    Ok(())
}

/// MXFP8-KV flash-decode (block-scale): `q`/`out` — `q_dtype` (bf16/f16/f32);
/// `k`/`v` — MXFP8 E4M3 (1 байт/элем), `k_scale`/`v_scale` — U8 E8M0 `[B,NKV,T,D/32]`
/// (per-32-block, physical T-stride = `t_stride`). Деквант per-32-block inline,
/// merge переиспользуется (F32-partial → q_dtype). Требует `d % 32 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_mxfp8_u8(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    k_scale: &CudaSlice<u8>,
    ks_off: usize,
    v_scale: &CudaSlice<u8>,
    vs_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
    t_stride: u32,
    q_dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 || d == 0 {
        return Ok(());
    }
    if d % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("flash mxfp8: d={d} must be %32")));
    }
    validate(nh, nkv, d, split_k)?;
    let esz_q = (q_dtype.size_in_bits() / 8) as usize;
    let t_stride_eff = if t_stride > 0 { t_stride } else { t_kv } as usize;
    let nb = (d / 32) as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    // E8M0 scale — 1 байт на (T-row, 32-блок).
    let scale_n = (b as usize) * (nkv as usize) * t_stride_eff * nb;
    let rows = b * nh * t_q;
    let n_partials = (rows as usize) * (split_k as usize);
    let mut partial_acc = stream
        .alloc_zeros::<f32>(n_partials * d as usize)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_acc: {e:?}")))?;
    let mut partial_m = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_m: {e:?}")))?;
    let mut partial_l = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_l: {e:?}")))?;
    let (b_i, nh_i, nkv_i, tq_i, tkv_i, d_i, sk_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        t_q as i32,
        t_kv as i32,
        d as i32,
        split_k as i32,
    );
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    // smem split: q_sh[D] + acc_sh[D] + s_sh[TILE_KV] (БЕЗ vscale_sh — V-scale
    // читается per-32-block из DRAM).
    let smem = (2 * d + 128) * 4;
    let split_cfg = LaunchConfig {
        grid_dim: (rows, split_k, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let merge_cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let k_v = k.slice(k_off..k_off + kv_n);
    let v_v = v.slice(v_off..v_off + kv_n);
    let ks_v = k_scale.slice(ks_off..ks_off + scale_n);
    let vs_v = v_scale.slice(vs_off..vs_off + scale_n);

    macro_rules! run {
        ($t:ty, $split:expr, $merge:expr) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz_q)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash mxfp8: transmute q".into()))?
            };
            {
                let mut bld = stream.launch_builder($split);
                bld.arg(&q_v)
                    .arg(&k_v)
                    .arg(&v_v)
                    .arg(&ks_v)
                    .arg(&vs_v)
                    .arg(&mut partial_acc)
                    .arg(&mut partial_m)
                    .arg(&mut partial_l)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&tq_i)
                    .arg(&tkv_i)
                    .arg(&d_i)
                    .arg(&scale)
                    .arg(&causal_i)
                    .arg(&sk_i)
                    .arg(&ts_i);
                unsafe {
                    bld.launch(split_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash mxfp8 split: {e:?}"))
                    })?;
                }
            }
            {
                let mut o_s = out.slice_mut(out_off..out_off + q_n * esz_q);
                let mut o_v = unsafe {
                    o_s.transmute_mut::<$t>(q_n)
                        .ok_or_else(|| SynaptixError::Cuda("flash mxfp8: transmute out".into()))?
                };
                let mut bld = stream.launch_builder($merge);
                bld.arg(&partial_acc)
                    .arg(&partial_m)
                    .arg(&partial_l)
                    .arg(&mut o_v)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&tq_i)
                    .arg(&d_i)
                    .arg(&sk_i);
                unsafe {
                    bld.launch(merge_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash mxfp8 merge: {e:?}"))
                    })?;
                }
            }
        }};
    }

    match q_dtype {
        DType::F32 => run!(f32, &kernels.split_mxfp8_f32, &kernels.merge_f32),
        DType::F16 => run!(f16, &kernels.split_mxfp8_f16, &kernels.merge_f16),
        DType::BF16 => run!(bf16, &kernels.split_mxfp8_bf16, &kernels.merge_bf16),
        _ => return Err(SynaptixError::Unsupported("flash_decode_mxfp8_u8: q dtype")),
    }
    Ok(())
}

/// Device-Tkv вариант [`flash_decode_mxfp8_u8`] (CUDA-graph decode): активная
/// длина KV `*tkv_dev` device-резидентна (launch config от неё не зависит → один
/// граф валиден для всех позиций). Требует preallocated буфер (`t_stride > 0`).
#[allow(clippy::too_many_arguments)]
pub fn flash_decode_mxfp8_u8_dev(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    k_scale: &CudaSlice<u8>,
    ks_off: usize,
    v_scale: &CudaSlice<u8>,
    vs_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    tkv_dev: &CudaView<u32>,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
    t_stride: u32,
    q_dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || d == 0 {
        return Ok(());
    }
    if t_stride == 0 {
        return Err(SynaptixError::Cuda(
            "flash_decode_mxfp8_u8_dev: требуется preallocated буфер (t_stride > 0)".into(),
        ));
    }
    if d % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("flash mxfp8 dev: d={d} must be %32")));
    }
    validate(nh, nkv, d, split_k)?;
    let esz_q = (q_dtype.size_in_bits() / 8) as usize;
    let t_stride_eff = t_stride as usize;
    let nb = (d / 32) as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let scale_n = (b as usize) * (nkv as usize) * t_stride_eff * nb;
    let rows = b * nh * t_q;
    let n_partials = (rows as usize) * (split_k as usize);
    let mut partial_acc = stream
        .alloc_zeros::<f32>(n_partials * d as usize)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_acc: {e:?}")))?;
    let mut partial_m = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_m: {e:?}")))?;
    let mut partial_l = stream
        .alloc_zeros::<f32>(n_partials)
        .map_err(|e| SynaptixError::Cuda(format!("alloc partial_l: {e:?}")))?;
    let (b_i, nh_i, nkv_i, tq_i, d_i, sk_i) =
        (b as i32, nh as i32, nkv as i32, t_q as i32, d as i32, split_k as i32);
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    let smem = (2 * d + 128) * 4;
    let split_cfg = LaunchConfig {
        grid_dim: (rows, split_k, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let merge_cfg = LaunchConfig {
        grid_dim: (rows, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };
    let k_v = k.slice(k_off..k_off + kv_n);
    let v_v = v.slice(v_off..v_off + kv_n);
    let ks_v = k_scale.slice(ks_off..ks_off + scale_n);
    let vs_v = v_scale.slice(vs_off..vs_off + scale_n);

    macro_rules! run {
        ($t:ty, $split:expr, $merge:expr) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz_q)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash mxfp8 dev: transmute q".into()))?
            };
            {
                let mut bld = stream.launch_builder($split);
                bld.arg(&q_v)
                    .arg(&k_v)
                    .arg(&v_v)
                    .arg(&ks_v)
                    .arg(&vs_v)
                    .arg(&mut partial_acc)
                    .arg(&mut partial_m)
                    .arg(&mut partial_l)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&nkv_i)
                    .arg(&tq_i)
                    .arg(tkv_dev)
                    .arg(&d_i)
                    .arg(&scale)
                    .arg(&causal_i)
                    .arg(&sk_i)
                    .arg(&ts_i);
                unsafe {
                    bld.launch(split_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash mxfp8 dev split: {e:?}"))
                    })?;
                }
            }
            {
                let mut o_s = out.slice_mut(out_off..out_off + q_n * esz_q);
                let mut o_v = unsafe {
                    o_s.transmute_mut::<$t>(q_n)
                        .ok_or_else(|| SynaptixError::Cuda("flash mxfp8 dev: transmute out".into()))?
                };
                let mut bld = stream.launch_builder($merge);
                bld.arg(&partial_acc)
                    .arg(&partial_m)
                    .arg(&partial_l)
                    .arg(&mut o_v)
                    .arg(&b_i)
                    .arg(&nh_i)
                    .arg(&tq_i)
                    .arg(&d_i)
                    .arg(&sk_i);
                unsafe {
                    bld.launch(merge_cfg).map_err(|e| {
                        SynaptixError::Cuda(format!("launch flash mxfp8 dev merge: {e:?}"))
                    })?;
                }
            }
        }};
    }

    match q_dtype {
        DType::F32 => run!(f32, &kernels.split_mxfp8_f32_dev, &kernels.merge_f32),
        DType::F16 => run!(f16, &kernels.split_mxfp8_f16_dev, &kernels.merge_f16),
        DType::BF16 => run!(bf16, &kernels.split_mxfp8_bf16_dev, &kernels.merge_bf16),
        _ => return Err(SynaptixError::Unsupported("flash_decode_mxfp8_u8_dev: q dtype")),
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn flash_decode_f32(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<f32>,
    k: &CudaSlice<f32>,
    v: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
) -> Result<()> {
    flash_decode::<f32>(
        kernels,
        stream,
        q,
        k,
        v,
        out,
        b,
        nh,
        nkv,
        t_q,
        t_kv,
        d,
        scale,
        causal,
        split_k,
        0,
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn flash_decode_f16(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<f16>,
    k: &CudaSlice<f16>,
    v: &CudaSlice<f16>,
    out: &mut CudaSlice<f16>,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
) -> Result<()> {
    flash_decode::<f16>(
        kernels,
        stream,
        q,
        k,
        v,
        out,
        b,
        nh,
        nkv,
        t_q,
        t_kv,
        d,
        scale,
        causal,
        split_k,
        0,
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn flash_decode_bf16(
    kernels: &FlashDecodeKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<bf16>,
    k: &CudaSlice<bf16>,
    v: &CudaSlice<bf16>,
    out: &mut CudaSlice<bf16>,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    t_kv: u32,
    d: u32,
    scale: f32,
    causal: bool,
    split_k: u32,
) -> Result<()> {
    flash_decode::<bf16>(
        kernels,
        stream,
        q,
        k,
        v,
        out,
        b,
        nh,
        nkv,
        t_q,
        t_kv,
        d,
        scale,
        causal,
        split_k,
        0,
        DType::BF16,
    )
}

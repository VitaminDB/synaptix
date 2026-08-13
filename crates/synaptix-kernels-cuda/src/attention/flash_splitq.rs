//! flash_splitq (sm_120) — split-Q FA-2-схема поверх mma.sync.m16n8k16.
//!
//! Единственное tensor-core prefill-ядро (flash_v4 удалён): BM=64 (4 warp'а × 16 q-строк),
//! online softmax в регистрах (shfl по четвёркам лейнов), P переиспользуется
//! из S-аккумуляторов как A-фрагменты PV-mma — без smem-roundtrip. На
//! [1,32,32640,128] bf16: v4 ~27 TFLOPS, цель FA-2/SDPA-класс (~93 TF).
//! Layout: BHSD + bshd-вариант, GQA, t_stride (префилл из KV-кэша).

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module_with_opts, load_fn};

const BM: u32 = 64;
const BN: u32 = 32;
const THREADS: u32 = 128;

pub struct FlashSplitQKernels {
    _module: Arc<CudaModule>,
    f16_hd64: CudaFunction,
    f16_hd128: CudaFunction,
    f16_hd256: CudaFunction,
    bf16_hd64: CudaFunction,
    bf16_hd128: CudaFunction,
    bf16_hd256: CudaFunction,
    f16_hd64_bshd: CudaFunction,
    bf16_hd64_bshd: CudaFunction,
    f16_hd128_bshd: CudaFunction,
    bf16_hd128_bshd: CudaFunction,
    f16_hd64_dev: CudaFunction,
    f16_hd128_dev: CudaFunction,
    f16_hd256_dev: CudaFunction,
    bf16_hd64_dev: CudaFunction,
    bf16_hd128_dev: CudaFunction,
    bf16_hd256_dev: CudaFunction,
    /// v5 (BM=64/BN=64/4 warps, split K/V commit): large-Tq HD=128 — 91.2 TF
    /// vs 84.7 у BN=32 на 14080² (карта тупиков свипа в flash_splitq.cu).
    f16_hd128_v5: CudaFunction,
    bf16_hd128_v5: CudaFunction,
    bf16_hd128_win: CudaFunction,
    bf16_hd128_v5_win: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashSplitQKernels>)>>> = OnceLock::new();

impl FlashSplitQKernels {
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
        let src = include_str!("../cu/fused/attention/flash_splitq.cu");
        let module = compile_module_with_opts(ctx, src, "flash_splitq.cu", &[], Some("sm_80"))?;
        let f16_hd64 = load_fn(&module, "flash_splitq_f16_hd64")?;
        let f16_hd128 = load_fn(&module, "flash_splitq_f16_hd128")?;
        let f16_hd256 = load_fn(&module, "flash_splitq_f16_hd256")?;
        let bf16_hd64 = load_fn(&module, "flash_splitq_bf16_hd64")?;
        let bf16_hd128 = load_fn(&module, "flash_splitq_bf16_hd128")?;
        let bf16_hd256 = load_fn(&module, "flash_splitq_bf16_hd256")?;
        let f16_hd64_bshd = load_fn(&module, "flash_splitq_f16_hd64_bshd")?;
        let bf16_hd64_bshd = load_fn(&module, "flash_splitq_bf16_hd64_bshd")?;
        let f16_hd128_bshd = load_fn(&module, "flash_splitq_f16_hd128_bshd")?;
        let bf16_hd128_bshd = load_fn(&module, "flash_splitq_bf16_hd128_bshd")?;
        let f16_hd64_dev = load_fn(&module, "flash_splitq_f16_hd64_dev")?;
        let f16_hd128_dev = load_fn(&module, "flash_splitq_f16_hd128_dev")?;
        let f16_hd256_dev = load_fn(&module, "flash_splitq_f16_hd256_dev")?;
        let bf16_hd64_dev = load_fn(&module, "flash_splitq_bf16_hd64_dev")?;
        let bf16_hd128_dev = load_fn(&module, "flash_splitq_bf16_hd128_dev")?;
        let bf16_hd256_dev = load_fn(&module, "flash_splitq_bf16_hd256_dev")?;
        let f16_hd128_v5 = load_fn(&module, "flash_splitq5_f16_hd128")?;
        let bf16_hd128_v5 = load_fn(&module, "flash_splitq5_bf16_hd128")?;
        let bf16_hd128_win = load_fn(&module, "flash_splitq_bf16_hd128_win")?;
        let bf16_hd128_v5_win = load_fn(&module, "flash_splitq5_bf16_hd128_win")?;

        // v5: smem = 2 (K,V) × 64 × 136 × 2 = 34816 Б < 48 KB, но ставим с запасом.
        for func in [&f16_hd128_v5, &bf16_hd128_v5, &bf16_hd128_v5_win] {
            func.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                96 * 1024,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attribute flash_splitq2 shared: {e:?}")))?;
        }
        // HD=256: smem = (64+2·32)·256·2 = 64 KB → выше дефолтных 48 KB.
        // `_dev`-варианты считают smem как 4·BN·(D+8)·2 = 66 KB при HD=256 —
        // им лимит нужен так же, иначе launch падает с CUDA_ERROR_INVALID_VALUE.
        for func in [
            &f16_hd64,
            &f16_hd128,
            &f16_hd256,
            &bf16_hd64,
            &bf16_hd128,
            &bf16_hd256,
            &f16_hd64_bshd,
            &bf16_hd64_bshd,
            &f16_hd128_bshd,
            &bf16_hd128_bshd,
            &bf16_hd128_win,
            &f16_hd64_dev,
            &f16_hd128_dev,
            &f16_hd256_dev,
            &bf16_hd64_dev,
            &bf16_hd128_dev,
            &bf16_hd256_dev,
        ] {
            func.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                80 * 1024,
            )
            .map_err(|e| SynaptixError::Cuda(format!("set_attribute flash_splitq shared: {e:?}")))?;
        }
        let new = Arc::new(Self {
            f16_hd64,
            f16_hd128,
            f16_hd256,
            bf16_hd64,
            bf16_hd128,
            bf16_hd256,
            f16_hd64_bshd,
            bf16_hd64_bshd,
            f16_hd128_bshd,
            bf16_hd128_bshd,
            f16_hd64_dev,
            f16_hd128_dev,
            f16_hd256_dev,
            bf16_hd64_dev,
            bf16_hd128_dev,
            bf16_hd256_dev,
            f16_hd128_v5,
            bf16_hd128_v5,
            bf16_hd128_win,
            bf16_hd128_v5_win,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn splitq_pick_func_dev(
    kernels: &FlashSplitQKernels,
    dtype: DType,
    d: u32,
) -> Result<&CudaFunction> {
    Ok(match (dtype, d) {
        (DType::F16, 64) => &kernels.f16_hd64_dev,
        (DType::F16, 128) => &kernels.f16_hd128_dev,
        (DType::F16, 256) => &kernels.f16_hd256_dev,
        (DType::BF16, 64) => &kernels.bf16_hd64_dev,
        (DType::BF16, 128) => &kernels.bf16_hd128_dev,
        (DType::BF16, 256) => &kernels.bf16_hd256_dev,
        _ => {
            return Err(SynaptixError::Unsupported(
                "flash_splitq_dev: HD in {64,128,256}, F16/BF16",
            ))
        }
    })
}

fn splitq_pick_func(
    kernels: &FlashSplitQKernels,
    dtype: DType,
    d: u32,
    bshd: bool,
) -> Result<&CudaFunction> {
    if bshd {
        return match (dtype, d) {
            (DType::F16, 64) => Ok(&kernels.f16_hd64_bshd),
            (DType::BF16, 64) => Ok(&kernels.bf16_hd64_bshd),
            (DType::F16, 128) => Ok(&kernels.f16_hd128_bshd),
            (DType::BF16, 128) => Ok(&kernels.bf16_hd128_bshd),
            _ => Err(SynaptixError::Unsupported("flash_splitq bshd: only HD=64 F16/BF16")),
        };
    }
    Ok(match (dtype, d) {
        (DType::F16, 64) => &kernels.f16_hd64,
        (DType::F16, 128) => &kernels.f16_hd128,
        (DType::F16, 256) => &kernels.f16_hd256,
        (DType::BF16, 64) => &kernels.bf16_hd64,
        (DType::BF16, 128) => &kernels.bf16_hd128,
        (DType::BF16, 256) => &kernels.bf16_hd256,
        (DType::F16 | DType::BF16, other) => {
            return Err(SynaptixError::Cuda(format!(
                "flash_splitq: HD must be 64, 128 or 256, got {other}"
            )))
        }
        (other, _) => {
            return Err(SynaptixError::Cuda(format!(
                "flash_splitq: unsupported dtype {other:?} (tensor-core: F16/BF16)"
            )))
        }
    })
}

// smem: double-buffer k+v 2×[2×BN][d+8] в T (2 байта); +8 эл./строку — анти-
// bank-conflict паддинг (flash_splitq.cu KV_LD). Q не стейджится (q_frag из global).
fn splitq_cfg(b: u32, nh: u32, t_q: u32, d: u32) -> LaunchConfig {
    let shared_bytes: u32 = 4 * BN * (d + 8) * 2;
    LaunchConfig {
        // X = q_tile (быстрейшая), Y = bh — q-тайлы одной головы соседние → K/V из L2.
        grid_dim: (t_q.div_ceil(BM), b * nh, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: shared_bytes,
    }
}

/// FA-5 из untyped `u8`-storage. HD ∈ {64,128,256}.
#[allow(clippy::too_many_arguments)]
pub fn flash_splitq_u8(
    kernels: &FlashSplitQKernels,
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
    t_stride: u32,
    dtype: DType,
    bshd: bool,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_splitq: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    // v5 (BM=64/BN=64, split K/V commit) для large-Tq HD=128.
    let use_v5 = !bshd && d == 128 && t_q >= 1024;
    let func = if use_v5 {
        match dtype {
            DType::F16 => &kernels.f16_hd128_v5,
            DType::BF16 => &kernels.bf16_hd128_v5,
            _ => return Err(SynaptixError::Unsupported("flash_splitq_u8: dtype (F16/BF16)")),
        }
    } else {
        splitq_pick_func(kernels, dtype, d, bshd)?
    };
    let cfg = if use_v5 {
        LaunchConfig {
            grid_dim: (t_q.div_ceil(BM), b * nh, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 2 * 64 * (d + 8) * 2,
        }
    } else {
        splitq_cfg(b, nh, t_q, d)
    };
    let esz = (dtype.size_in_bits() / 8) as usize;
    let t_stride_eff = if t_stride > 0 { t_stride } else { t_kv } as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let (b_i, nh_i, nkv_i, tq_i, tkv_i) =
        (b as i32, nh as i32, nkv as i32, t_q as i32, t_kv as i32);
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    macro_rules! run {
        ($t:ty) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq u8: transmute q".into()))?
            };
            let k_v = unsafe {
                k.slice(k_off..k_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq u8: transmute k".into()))?
            };
            let v_v = unsafe {
                v.slice(v_off..v_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq u8: transmute v".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq u8: transmute out".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&mut o_v)
                .arg(&scale)
                .arg(&b_i)
                .arg(&nh_i)
                .arg(&nkv_i)
                .arg(&tq_i)
                .arg(&tkv_i)
                .arg(&causal_i)
                .arg(&ts_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch flash_splitq u8: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => return Err(SynaptixError::Unsupported("flash_splitq_u8: dtype (F16/BF16)")),
    }
    Ok(())
}

/// Bidirectional sliding-window FA-5 (bf16, HD=128). `window>0` → key j виден
/// строке i только при |i-j|<=window (band-маска + скип kv-блоков вне окна).
#[allow(clippy::too_many_arguments)]
pub fn flash_splitq_window_u8(
    kernels: &FlashSplitQKernels,
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
    scale: f32,
    causal: bool,
    t_stride: u32,
    window: i32,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_splitq_window: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    let d: u32 = 128;
    let use_v5 = t_q >= 1024;
    let func = if use_v5 {
        &kernels.bf16_hd128_v5_win
    } else {
        &kernels.bf16_hd128_win
    };
    let cfg = if use_v5 {
        LaunchConfig {
            grid_dim: (t_q.div_ceil(BM), b * nh, 1),
            block_dim: (THREADS, 1, 1),
            shared_mem_bytes: 2 * 64 * (d + 8) * 2,
        }
    } else {
        splitq_cfg(b, nh, t_q, d)
    };
    let esz = 2usize;
    let t_stride_eff = if t_stride > 0 { t_stride } else { t_kv } as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let (b_i, nh_i, nkv_i, tq_i, tkv_i) =
        (b as i32, nh as i32, nkv as i32, t_q as i32, t_kv as i32);
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    let q_v = unsafe {
        q.slice(q_off..q_off + q_n * esz)
            .transmute::<bf16>(q_n)
            .ok_or_else(|| SynaptixError::Cuda("flash_splitq_window: transmute q".into()))?
    };
    let k_v = unsafe {
        k.slice(k_off..k_off + kv_n * esz)
            .transmute::<bf16>(kv_n)
            .ok_or_else(|| SynaptixError::Cuda("flash_splitq_window: transmute k".into()))?
    };
    let v_v = unsafe {
        v.slice(v_off..v_off + kv_n * esz)
            .transmute::<bf16>(kv_n)
            .ok_or_else(|| SynaptixError::Cuda("flash_splitq_window: transmute v".into()))?
    };
    let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
    let mut o_v = unsafe {
        o_s.transmute_mut::<bf16>(q_n)
            .ok_or_else(|| SynaptixError::Cuda("flash_splitq_window: transmute out".into()))?
    };
    let mut bld = stream.launch_builder(func);
    bld.arg(&q_v)
        .arg(&k_v)
        .arg(&v_v)
        .arg(&mut o_v)
        .arg(&scale)
        .arg(&b_i)
        .arg(&nh_i)
        .arg(&nkv_i)
        .arg(&tq_i)
        .arg(&tkv_i)
        .arg(&causal_i)
        .arg(&ts_i)
        .arg(&window);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch flash_splitq_window: {e:?}")))?;
    }
    Ok(())
}

/// Как [`flash_splitq_u8`], но активная длина KV читается ядром из device-буфера
/// `tcache` (i32) — для CUDA-graph prefill (грид статичен по Tq). Контракт =
/// host-варианта, но Tkv читается ядром из device-буфера.
#[allow(clippy::too_many_arguments)]
pub fn flash_splitq_u8_dev(
    kernels: &FlashSplitQKernels,
    stream: &Arc<CudaStream>,
    q: &CudaSlice<u8>,
    q_off: usize,
    k: &CudaSlice<u8>,
    k_off: usize,
    v: &CudaSlice<u8>,
    v_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    tcache: &CudaSlice<u8>,
    tc_off: usize,
    b: u32,
    nh: u32,
    nkv: u32,
    t_q: u32,
    d: u32,
    scale: f32,
    causal: bool,
    t_stride: u32,
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_splitq_dev: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    if t_stride == 0 {
        return Err(SynaptixError::Cuda(
            "flash_splitq_dev: t_stride must be > 0 (device-resident-Tkv requires preallocated KV view)".into(),
        ));
    }
    let func = splitq_pick_func_dev(kernels, dtype, d)?;
    let cfg = splitq_cfg(b, nh, t_q, d);
    let esz = (dtype.size_in_bits() / 8) as usize;
    let t_stride_eff = t_stride as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let (b_i, nh_i, nkv_i, tq_i) = (b as i32, nh as i32, nkv as i32, t_q as i32);
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    let tc_view = unsafe {
        tcache
            .slice(tc_off..tc_off + 4)
            .transmute::<i32>(1)
            .ok_or_else(|| SynaptixError::Cuda("flash_splitq_dev: transmute tcache".into()))?
    };
    macro_rules! run {
        ($t:ty) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq_dev: transmute q".into()))?
            };
            let k_v = unsafe {
                k.slice(k_off..k_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq_dev: transmute k".into()))?
            };
            let v_v = unsafe {
                v.slice(v_off..v_off + kv_n * esz)
                    .transmute::<$t>(kv_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq_dev: transmute v".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_splitq_dev: transmute out".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&mut o_v)
                .arg(&scale)
                .arg(&b_i)
                .arg(&nh_i)
                .arg(&nkv_i)
                .arg(&tq_i)
                .arg(&tc_view)
                .arg(&causal_i)
                .arg(&ts_i);
            unsafe {
                bld.launch(cfg)
                    .map_err(|e| SynaptixError::Cuda(format!("launch flash_splitq_dev: {e:?}")))?;
            }
        }};
    }
    match dtype {
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => {
            return Err(SynaptixError::Unsupported(
                "flash_splitq_u8_dev: dtype (F16/BF16)",
            ))
        }
    }
    Ok(())
}

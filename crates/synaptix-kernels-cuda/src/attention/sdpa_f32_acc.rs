//! Наивный SDPA с полной F32-аккумуляцией — точный baseline/reference (НЕ flash).
//!
//! `out = softmax(scale · Q·Kᵀ + causal_mask) · V`. GQA (NH кратно NKV), causal.
//! Один block = одна q-позиция, scores материализуются в dynamic shared memory
//! (лимит `Tkv·4 ≤ 48 КБ`). F32/F16/BF16, аккумулятор F32. Layout совпадает с
//! flash_v2: q (B, NH, Tq, D), k/v (B, NKV, Tkv, D), out (B, NH, Tq, D).

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

const BLOCK: u32 = 128;
const MAX_SMEM_BYTES: u32 = 48 * 1024;

pub struct SdpaF32AccKernels {
    _module: Arc<CudaModule>,
    f32: CudaFunction,
    f16: CudaFunction,
    bf16: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<SdpaF32AccKernels>)>>> = OnceLock::new();

impl SdpaF32AccKernels {
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
        let src = include_str!("../cu/fused/attention/sdpa_f32_acc.cu");
        let module = compile_module(ctx, src, "sdpa_f32_acc.cu")?;
        let new = Arc::new(Self {
            f32: load_fn(&module, "sdpa_f32_acc_f32")?,
            f16: load_fn(&module, "sdpa_f32_acc_f16")?,
            bf16: load_fn(&module, "sdpa_f32_acc_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

/// `out = softmax(scale·Q·Kᵀ + causal_mask)·V`. См. layout в модуле.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_f32_acc<T: DeviceRepr>(
    kernels: &SdpaF32AccKernels,
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
    dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 || d == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "sdpa_f32_acc: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    let smem = t_kv
        .checked_mul(4)
        .ok_or_else(|| SynaptixError::Cuda("sdpa_f32_acc: Tkv overflow".to_string()))?;
    if smem > MAX_SMEM_BYTES {
        return Err(SynaptixError::Cuda(format!(
            "sdpa_f32_acc: Tkv={t_kv} (scores {smem}B) превышает лимит shared memory {MAX_SMEM_BYTES}B"
        )));
    }
    let func = match dtype {
        DType::F32 => &kernels.f32,
        DType::F16 => &kernels.f16,
        DType::BF16 => &kernels.bf16,
        other => {
            return Err(SynaptixError::Cuda(format!(
                "sdpa_f32_acc: unsupported dtype {other:?}"
            )))
        }
    };
    let cfg = LaunchConfig {
        grid_dim: (b * nh * t_q, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: smem,
    };
    let (b_i, nh_i, nkv_i, tq_i, tkv_i, d_i) = (
        b as i32,
        nh as i32,
        nkv as i32,
        t_q as i32,
        t_kv as i32,
        d as i32,
    );
    let causal_i: i32 = if causal { 1 } else { 0 };
    let mut bld = stream.launch_builder(func);
    bld.arg(q)
        .arg(k)
        .arg(v)
        .arg(&mut *out)
        .arg(&b_i)
        .arg(&nh_i)
        .arg(&nkv_i)
        .arg(&tq_i)
        .arg(&tkv_i)
        .arg(&d_i)
        .arg(&scale)
        .arg(&causal_i);
    unsafe {
        bld.launch(cfg)
            .map_err(|e| SynaptixError::Cuda(format!("launch sdpa_f32_acc: {e:?}")))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn sdpa_f32_acc_f32(
    kernels: &SdpaF32AccKernels,
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
) -> Result<()> {
    sdpa_f32_acc::<f32>(
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
        DType::F32,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sdpa_f32_acc_f16(
    kernels: &SdpaF32AccKernels,
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
) -> Result<()> {
    sdpa_f32_acc::<f16>(
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
        DType::F16,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn sdpa_f32_acc_bf16(
    kernels: &SdpaF32AccKernels,
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
) -> Result<()> {
    sdpa_f32_acc::<bf16>(
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
        DType::BF16,
    )
}

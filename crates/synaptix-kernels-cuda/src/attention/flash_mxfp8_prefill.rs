//! MXFP8-KV prefill attention: K/V — E4M3 + per-32-block E8M0 scale, Q/out —
//! F16/BF16. Block-деквант в smem + tensor-core MMA. Унаследован от удалённого
//! flash_v4 (BM=16, серийный softmax) — СТРУКТУРНО МЕДЛЕННЫЙ на больших S.
//! TODO: портировать на схему flash_splitq (split-Q, softmax в регистрах).

use std::sync::{Arc, OnceLock};

use cudarc::driver::sys::CUfunction_attribute_enum;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use half::{bf16, f16};
use parking_lot::Mutex;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

const BM: u32 = 16;
const BN: u32 = 32;
const BLOCK_D: u32 = 128;

pub struct FlashMxfp8PrefillKernels {
    f16_hd128: CudaFunction,
    f16_hd256: CudaFunction,
    bf16_hd128: CudaFunction,
    bf16_hd256: CudaFunction,
    _module: Arc<CudaModule>,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<FlashMxfp8PrefillKernels>)>>> = OnceLock::new();

impl FlashMxfp8PrefillKernels {
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
        let src = include_str!("../cu/fused/attention/flash_mxfp8_prefill.cu");
        let module = compile_module(ctx, src, "flash_mxfp8_prefill.cu")?;
        let f16_hd128 = load_fn(&module, "flash_mxfp8_f16_hd128")?;
        let f16_hd256 = load_fn(&module, "flash_mxfp8_f16_hd256")?;
        let bf16_hd128 = load_fn(&module, "flash_mxfp8_bf16_hd128")?;
        let bf16_hd256 = load_fn(&module, "flash_mxfp8_bf16_hd256")?;
        for func in [&f16_hd128, &f16_hd256, &bf16_hd128, &bf16_hd256] {
            func.set_attribute(
                CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                80 * 1024,
            )
            .map_err(|e| {
                SynaptixError::Cuda(format!("set_attribute flash_mxfp8 shared: {e:?}"))
            })?;
        }
        let new = Arc::new(Self {
            f16_hd128,
            f16_hd256,
            bf16_hd128,
            bf16_hd256,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn mxfp8_cfg(b: u32, nh: u32, t_q: u32, d: u32) -> LaunchConfig {
    let shared_bytes: u32 = BM * d * 2 + 2 * BN * d * 2 + BM * BN * 4 + BM * BN * 2 + 3 * BM * 4;
    LaunchConfig {
        grid_dim: (b * nh, t_q.div_ceil(BM), 1),
        block_dim: (BLOCK_D, 1, 1),
        shared_mem_bytes: shared_bytes,
    }
}

fn mxfp8_pick_func(kernels: &FlashMxfp8PrefillKernels, dtype: DType, d: u32) -> Result<&CudaFunction> {
    Ok(match (dtype, d) {
        (DType::F16, 128) => &kernels.f16_hd128,
        (DType::F16, 256) => &kernels.f16_hd256,
        (DType::BF16, 128) => &kernels.bf16_hd128,
        (DType::BF16, 256) => &kernels.bf16_hd256,
        (dt, other) => {
            return Err(SynaptixError::Cuda(format!(
                "flash_mxfp8: unsupported dtype {dt:?} / HD {other} (F16/BF16 × 128/256)"
            )))
        }
    })
}

/// MXFP8-KV prefill (tensor-core) из untyped storage. `q`/`out` — `q_dtype`
/// (F16/BF16); `k`/`v` — MXFP8 E4M3, `k_scale`/`v_scale` — U8 E8M0
/// `[B,NKV,Tkv,D/32]` (per-32-block, physical T-stride = `t_stride`). Деквант
/// block-scale при загрузке в smem, далее tensor-core MMA. HD ∈ {128,256}, D%32==0.
#[allow(clippy::too_many_arguments)]
pub fn flash_mxfp8_prefill_u8(
    kernels: &FlashMxfp8PrefillKernels,
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
    t_stride: u32,
    q_dtype: DType,
) -> Result<()> {
    if b == 0 || nh == 0 || t_q == 0 || t_kv == 0 {
        return Ok(());
    }
    if nkv == 0 || nh % nkv != 0 {
        return Err(SynaptixError::Cuda(format!(
            "flash_mxfp8: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    if d % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("flash_mxfp8: D={d} must be %32")));
    }
    let func = mxfp8_pick_func(kernels, q_dtype, d)?;
    let cfg = mxfp8_cfg(b, nh, t_q, d);
    let esz = (q_dtype.size_in_bits() / 8) as usize;
    let t_stride_eff = if t_stride > 0 { t_stride } else { t_kv } as usize;
    let nb = (d / 32) as usize;
    let q_n = (b as usize) * (nh as usize) * (t_q as usize) * (d as usize);
    let kv_n = (b as usize) * (nkv as usize) * t_stride_eff * (d as usize);
    let scale_n = (b as usize) * (nkv as usize) * t_stride_eff * nb;
    let (b_i, nh_i, nkv_i, tq_i, tkv_i) =
        (b as i32, nh as i32, nkv as i32, t_q as i32, t_kv as i32);
    let causal_i: i32 = if causal { 1 } else { 0 };
    let ts_i: i32 = t_stride as i32;
    let k_v = k.slice(k_off..k_off + kv_n);
    let v_v = v.slice(v_off..v_off + kv_n);
    let ks_v = k_scale.slice(ks_off..ks_off + scale_n);
    let vs_v = v_scale.slice(vs_off..vs_off + scale_n);
    macro_rules! run {
        ($t:ty) => {{
            let q_v = unsafe {
                q.slice(q_off..q_off + q_n * esz)
                    .transmute::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_mxfp8: transmute q".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_mxfp8: transmute out".into()))?
            };
            let mut bld = stream.launch_builder(func);
            bld.arg(&q_v)
                .arg(&k_v)
                .arg(&v_v)
                .arg(&ks_v)
                .arg(&vs_v)
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
                    .map_err(|e| SynaptixError::Cuda(format!("launch flash_mxfp8 u8: {e:?}")))?;
            }
        }};
    }
    match q_dtype {
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => {
            return Err(SynaptixError::Unsupported(
                "flash_mxfp8_prefill_u8: q dtype (F16/BF16)",
            ))
        }
    }
    Ok(())
}

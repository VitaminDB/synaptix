//! MXFP8-KV prefill v2 (sm_120a): схема flash_splitq (FA-2 split-Q,
//! mma.sync.m16n8k16, online softmax в регистрах) с деквант-fill E4M3+E8M0 →
//! T прямо в smem аппаратным cvt (см. `flash_mxfp8_splitq.cu`). Заменяет
//! структурно медленный `flash_mxfp8_prefill` (BM=16, серийный softmax).
//!
//! Модуль sm_120a: на старых архитектурах загрузка падает,
//! [`FlashMxfp8SplitqKernels::try_for_context`] возвращает `None`, диспетчер
//! остаётся на WMMA-пути. Ограничения: HD ∈ {128,256}, q/out — F16/BF16,
//! k/v-байты выровнены на 16 (uint4-загрузки).

use std::sync::{Arc, OnceLock};

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

pub struct FlashMxfp8SplitqKernels {
    f16_hd128: CudaFunction,
    f16_hd256: CudaFunction,
    bf16_hd128: CudaFunction,
    bf16_hd256: CudaFunction,
    _module: Arc<CudaModule>,
}

/// Кэш по контексту; `None` — компиляция/загрузка не удалась (не sm_120a).
static CACHE: OnceLock<Mutex<Vec<(usize, Option<Arc<FlashMxfp8SplitqKernels>>)>>> =
    OnceLock::new();

impl FlashMxfp8SplitqKernels {
    pub fn try_for_context(ctx: &Arc<CudaContext>) -> Option<Arc<Self>> {
        let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
        let key = Arc::as_ptr(ctx) as usize;
        {
            let g = cache.lock();
            for (k, v) in g.iter() {
                if *k == key {
                    return v.clone();
                }
            }
        }
        let built = Self::build(ctx)
            .map_err(|e| {
                eprintln!("[flash_mxfp8_splitq] недоступен, fallback на WMMA-путь: {e}");
                e
            })
            .ok();
        cache.lock().push((key, built.clone()));
        built
    }

    fn build(ctx: &Arc<CudaContext>) -> Result<Arc<Self>> {
        let src = include_str!("../cu/fused/attention/flash_mxfp8_splitq.cu");
        let module =
            compile_module_with_opts(ctx, src, "flash_mxfp8_splitq.cu", &[], Some("sm_120a"))?;
        Ok(Arc::new(Self {
            f16_hd128: load_fn(&module, "flash_mxfp8_splitq_f16_hd128")?,
            f16_hd256: load_fn(&module, "flash_mxfp8_splitq_f16_hd256")?,
            bf16_hd128: load_fn(&module, "flash_mxfp8_splitq_bf16_hd128")?,
            bf16_hd256: load_fn(&module, "flash_mxfp8_splitq_bf16_hd256")?,
            _module: module,
        }))
    }

    fn pick(&self, dtype: DType, d: u32) -> Result<&CudaFunction> {
        Ok(match (dtype, d) {
            (DType::F16, 128) => &self.f16_hd128,
            (DType::F16, 256) => &self.f16_hd256,
            (DType::BF16, 128) => &self.bf16_hd128,
            (DType::BF16, 256) => &self.bf16_hd256,
            (dt, other) => {
                return Err(SynaptixError::Cuda(format!(
                    "flash_mxfp8_splitq: dtype {dt:?} / HD {other} (F16/BF16 × 128/256)"
                )))
            }
        })
    }
}

/// Пригодность формы/типов (без учёта наличия sm_120a-модуля).
pub fn splitq_shape_ok(d: u32, q_dtype: DType, k_off: usize, v_off: usize) -> bool {
    (d == 128 || d == 256)
        && matches!(q_dtype, DType::F16 | DType::BF16)
        && k_off % 16 == 0
        && v_off % 16 == 0
}

/// MXFP8-KV prefill (split-Q tensor-core) из untyped storage — сигнатура
/// зеркалит [`super::flash_mxfp8_prefill::flash_mxfp8_prefill_u8`].
#[allow(clippy::too_many_arguments)]
pub fn flash_mxfp8_splitq_u8(
    kernels: &FlashMxfp8SplitqKernels,
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
            "flash_mxfp8_splitq: NH={nh} must be a multiple of NKV={nkv}"
        )));
    }
    let func = kernels.pick(q_dtype, d)?;
    // KV-строки в smem с паддингом +8 эл. (ldmatrix-банки), T = 2 байта.
    let smem = 2 * BN * (d + 8) * 2;
    let cfg = LaunchConfig {
        grid_dim: (t_q.div_ceil(BM), b * nh, 1),
        block_dim: (THREADS, 1, 1),
        shared_mem_bytes: smem,
    };
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
                    .ok_or_else(|| SynaptixError::Cuda("flash_mxfp8_splitq: transmute q".into()))?
            };
            let mut o_s = out.slice_mut(out_off..out_off + q_n * esz);
            let mut o_v = unsafe {
                o_s.transmute_mut::<$t>(q_n)
                    .ok_or_else(|| SynaptixError::Cuda("flash_mxfp8_splitq: transmute out".into()))?
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
                bld.launch(cfg).map_err(|e| {
                    SynaptixError::Cuda(format!("launch flash_mxfp8_splitq: {e:?}"))
                })?;
            }
        }};
    }
    match q_dtype {
        DType::F16 => run!(f16),
        DType::BF16 => run!(bf16),
        _ => {
            return Err(SynaptixError::Unsupported(
                "flash_mxfp8_splitq_u8: q dtype (F16/BF16)",
            ))
        }
    }
    Ok(())
}

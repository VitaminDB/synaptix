//! Слитые ядра шага декода LLM (см. `cu/fused/llm_decode.cu`): хвост после
//! внимания, хвост FFN, роутер MoE с top-k, geglu с квантом, подготовка
//! внимания. Все аргументы-указатели приходят device-адресами (`u64`, 0 —
//! отсутствует): у ядер много необязательных входов, и типизированные вьюхи
//! только мешали бы.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg};
use parking_lot::Mutex;
use synaptix_core::error::{Result, SynaptixError};

use crate::kernels::compile::{compile_module, load_fn};

pub struct LlmDecodeKernels {
    _module: Arc<CudaModule>,
    attn_tail: CudaFunction,
    ffn_tail: CudaFunction,
    geglu_f16: CudaFunction,
    geglu_bf16: CudaFunction,
    attn_prep: CudaFunction,
}

static CACHE: OnceLock<Mutex<Vec<(usize, Arc<LlmDecodeKernels>)>>> = OnceLock::new();

impl LlmDecodeKernels {
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
        let src = include_str!("../cu/fused/llm_decode.cu");
        let module = compile_module(ctx, src, "llm_decode.cu")?;
        let new = Arc::new(Self {
            attn_tail: load_fn(&module, "dec_attn_tail_bf16")?,
            ffn_tail: load_fn(&module, "dec_ffn_tail_bf16")?,
            geglu_f16: load_fn(&module, "dec_geglu_quant_nvfp4_f16")?,
            geglu_bf16: load_fn(&module, "dec_geglu_quant_nvfp4_bf16")?,
            attn_prep: load_fn(&module, "dec_attn_prep_bf16")?,
            _module: module,
        });
        cache.lock().push((key, new.clone()));
        Ok(new)
    }
}

fn launch_err(name: &str, e: cudarc::driver::DriverError) -> SynaptixError {
    SynaptixError::Cuda(format!("launch {name}: {e:?}"))
}

/// Блок одной строки хвостов: по DEC_MAXE = 4 элемента на нить (строка живёт в
/// регистрах), до 1024 нитей.
fn row_block(h: u32) -> u32 {
    h.div_ceil(4).next_multiple_of(32).clamp(128, 1024)
}

const ROW_MAX_H: u32 = 4 * 1024;

/// NVFP4-выход нормы: адреса packed и scales (0 — выход выключен).
#[derive(Clone, Copy, Default)]
pub struct Nvfp4Out {
    pub w: u64,
    pub packed: u64,
    pub scales: u64,
}

/// Хвост после внимания, см. `dec_attn_tail_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn attn_tail(
    k: &LlmDecodeKernels,
    stream: &Arc<CudaStream>,
    attn_out: u64,
    post_w: u64,
    hidden_in: u64,
    hidden_out: u64,
    a: Nvfp4Out,
    b: Nvfp4Out,
    w_c: u64,
    c_out: u64,
    h: u32,
    eps_post: f32,
    eps: f32,
) -> Result<()> {
    if h % 16 != 0 {
        return Err(SynaptixError::Cuda(format!("attn_tail: H={h} кратно 16")));
    }
    if h > ROW_MAX_H {
        return Err(SynaptixError::Cuda(format!("attn_tail: H={h} > {ROW_MAX_H}")));
    }
    let hi = h as i32;
    let sf_inner = (h.div_ceil(64) * 4) as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (row_block(h), 1, 1),
        shared_mem_bytes: h * 4,
    };
    let mut bld = stream.launch_builder(&k.attn_tail);
    bld.arg(&attn_out)
        .arg(&post_w)
        .arg(&hidden_in)
        .arg(&hidden_out)
        .arg(&a.w)
        .arg(&a.packed)
        .arg(&a.scales)
        .arg(&b.w)
        .arg(&b.packed)
        .arg(&b.scales)
        .arg(&w_c)
        .arg(&c_out)
        .arg(&hi)
        .arg(&eps_post)
        .arg(&eps)
        .arg(&sf_inner);
    unsafe { bld.launch(cfg) }.map_err(|e| launch_err("dec_attn_tail", e)).map(|_| ())
}

/// Хвост FFN-части блока, см. `dec_ffn_tail_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn ffn_tail(
    k: &LlmDecodeKernels,
    stream: &Arc<CudaStream>,
    dense_out: u64,
    moe_acc: u64,
    hidden_in: u64,
    w_post_dense: u64,
    w_post_moe: u64,
    w_post_mlp: u64,
    layer_scalar: f32,
    hidden_out: u64,
    w_next: u64,
    next_bf16: u64,
    next_mx_packed: u64,
    next_mx_scales: u64,
    h: u32,
    eps_post: f32,
    eps: f32,
) -> Result<()> {
    if h % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("ffn_tail: H={h} кратно 32")));
    }
    if h > ROW_MAX_H {
        return Err(SynaptixError::Cuda(format!("ffn_tail: H={h} > {ROW_MAX_H}")));
    }
    let hi = h as i32;
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (row_block(h), 1, 1),
        shared_mem_bytes: h * 4,
    };
    let mut bld = stream.launch_builder(&k.ffn_tail);
    bld.arg(&dense_out)
        .arg(&moe_acc)
        .arg(&hidden_in)
        .arg(&w_post_dense)
        .arg(&w_post_moe)
        .arg(&w_post_mlp)
        .arg(&layer_scalar)
        .arg(&hidden_out)
        .arg(&w_next)
        .arg(&next_bf16)
        .arg(&next_mx_packed)
        .arg(&next_mx_scales)
        .arg(&hi)
        .arg(&eps_post)
        .arg(&eps);
    unsafe { bld.launch(cfg) }.map_err(|e| launch_err("dec_ffn_tail", e)).map(|_| ())
}

/// top-k роутера MoE, выполняемый последним блоком ядра geglu (device-адреса;
/// `logits` f32 [e], `pes` f32 [e] | 0, выходы `idx` u32 [k], `w` f32 [k],
/// `acc_zero` f32 [h] обнуляется).
#[derive(Clone, Copy, Default)]
pub struct TopkArgs {
    pub logits: u64,
    pub pes: u64,
    pub idx: u64,
    pub w: u64,
    pub acc_zero: u64,
    pub e: u32,
    pub k: u32,
    pub h: u32,
}

/// gelu_tanh(gate)·up → NVFP4-пара строк, см. `dec_geglu_quant_nvfp4_*`.
/// `stride` — шаг строки в элементах у gate и up. `topk.e > 0` добавляет блок
/// с top-k роутера.
#[allow(clippy::too_many_arguments)]
pub fn geglu_quant_nvfp4(
    k: &LlmDecodeKernels,
    stream: &Arc<CudaStream>,
    bf16: bool,
    gate: u64,
    up: u64,
    stride: u64,
    packed: u64,
    scales: u64,
    rows: u32,
    inter: u32,
    topk: TopkArgs,
) -> Result<()> {
    if inter % 16 != 0 {
        return Err(SynaptixError::Cuda(format!("geglu_quant: I={inter} кратно 16")));
    }
    if topk.e > 1024 || topk.k > 32 {
        return Err(SynaptixError::Cuda(format!("geglu_quant: top-k e={} ≤ 1024, k={} ≤ 32", topk.e, topk.k)));
    }
    let groups = rows * (inter / 16) * 4;
    if groups == 0 {
        return Ok(());
    }
    let block = 128u32;
    let (ri, ii) = (rows as i32, inter as i32);
    let sf_inner = (inter.div_ceil(64) * 4) as i32;
    let stride_i = stride as i64;
    let extra = u32::from(topk.e > 0);
    let cfg = LaunchConfig {
        grid_dim: (groups.div_ceil(block) + extra, 1, 1),
        block_dim: (block, 1, 1),
        shared_mem_bytes: 0,
    };
    let (te, tk, th) = (topk.e as i32, topk.k as i32, topk.h as i32);
    let f = if bf16 { &k.geglu_bf16 } else { &k.geglu_f16 };
    let mut bld = stream.launch_builder(f);
    bld.arg(&gate)
        .arg(&up)
        .arg(&stride_i)
        .arg(&packed)
        .arg(&scales)
        .arg(&ri)
        .arg(&ii)
        .arg(&sf_inner)
        .arg(&topk.logits)
        .arg(&topk.pes)
        .arg(&topk.idx)
        .arg(&topk.w)
        .arg(&topk.acc_zero)
        .arg(&te)
        .arg(&tk)
        .arg(&th);
    unsafe { bld.launch(cfg) }.map_err(|e| launch_err("dec_geglu_quant_nvfp4", e)).map(|_| ())
}

/// Нормы голов + RoPE + запись в KV, см. `dec_attn_prep_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn attn_prep(
    k: &LlmDecodeKernels,
    stream: &Arc<CudaStream>,
    q_in: u64,
    k_in: u64,
    v_in: u64,
    q_norm_w: u64,
    k_norm_w: u64,
    v_norm: bool,
    cos_t: u64,
    sin_t: u64,
    pos_ptr: u64,
    rotary_dim: u32,
    kv_pos_ptr: u64,
    q_out: u64,
    k_cache: u64,
    v_cache: u64,
    max_seq: u32,
    nh: u32,
    nkv: u32,
    hd: u32,
    eps: f32,
) -> Result<()> {
    if hd == 0 || hd > 1024 || hd % 32 != 0 {
        return Err(SynaptixError::Cuda(format!("attn_prep: hd={hd} (кратно 32, ≤1024)")));
    }
    let vn: i32 = i32::from(v_norm);
    let (rd, ms, nhi, nkvi, hdi) = (rotary_dim as i32, max_seq as i32, nh as i32, nkv as i32, hd as i32);
    let cfg = LaunchConfig {
        grid_dim: (nh + 2 * nkv, 1, 1),
        block_dim: (hd, 1, 1),
        shared_mem_bytes: hd * 4,
    };
    let mut bld = stream.launch_builder(&k.attn_prep);
    bld.arg(&q_in)
        .arg(&k_in)
        .arg(&v_in)
        .arg(&q_norm_w)
        .arg(&k_norm_w)
        .arg(&vn)
        .arg(&cos_t)
        .arg(&sin_t)
        .arg(&pos_ptr)
        .arg(&rd)
        .arg(&kv_pos_ptr)
        .arg(&q_out)
        .arg(&k_cache)
        .arg(&v_cache)
        .arg(&ms)
        .arg(&nhi)
        .arg(&nkvi)
        .arg(&hdi)
        .arg(&eps);
    unsafe { bld.launch(cfg) }.map_err(|e| launch_err("dec_attn_prep", e)).map(|_| ())
}

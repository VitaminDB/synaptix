//! Device-резидентный оркестратор chunked linear-attn prefill (T≥1).
//! Заменяет host-mix-блок `LinearAttn::forward` (synaptix-models/llm/common
//! model.rs:879-915): host_vec → causal_conv1d_stateful → silu → host_vec(a,b)
//! → gated_delta_decay_beta → scatter qe/ke/vv → gated_delta_rule_prefill.
//!
//! Цепочка:
//!   1. `causal_conv1d_chunk_<dtype>` (+ silu) — post_conv `[T, conv_dim]`
//!   2. `linear_attn_prep_scatter_<dtype>` — qe/ke/vv (F32) + g/β (F32)
//!   3. `chunk_gated_delta_rule` — обновляет ssm_state in-place, out `[num_v,T,hv]`
//!
//! Все промежуточные тензоры — owned scratch (alloc из mempool). Вход — typed
//! u8-views поверх Storage (как `linear_attn_decode_step_u8_dev`); transmute
//! по compute-dtype происходит внутри. Bit-exact с host-mix для F32; F16/BF16
//! compute даёт квантизационный дрейф ≤ tol каждого этапа (см. unit-тесты ядер).

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use parking_lot::Mutex;

use cudarc::driver::{CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};

use crate::attention::chunk_fla::ChunkFlaKernels;
use crate::attention::linear_attn_raw::LinearAttnRawKernels;
use crate::conv::causal_conv1d::{
    causal_conv1d_chunk_bf16, causal_conv1d_chunk_f16, causal_conv1d_chunk_f32, CausalConv1dKernels,
};
use crate::scan::chunk_scan::{chunk_gated_delta_rule, ChunkScanKernels};
use crate::wsalloc::{WsAlloc, WsBuf};

fn tm<T>() -> usize {
    std::mem::size_of::<T>()
}

/// Скретчи `linear_attn_chunk_prefill_u8_dev`, живущие между вызовами (набор
/// на устройство). Размеры зависят от (num_v, t_pad, hk, hv, conv_dim) — на
/// префилле они одинаковы для всех 48 linear-слоёв каждого чанка, поэтому
/// после первого слоя аллокаций не остаётся. Прежний вариант брал 8 свежих
/// блоков на слой на чанк и вместе с `chunk_scan` дробил пул в решето (OOM на
/// 6-МБ скретче при гигабайтах свободного внутри пула). Отдаются драйверу в
/// [`clear_linear_prefill_ws`] (из `release_device_caches`).
#[derive(Default)]
struct PrefillWs {
    qe: WsBuf<f32>,
    ke: WsBuf<f32>,
    vv: WsBuf<f32>,
    g_buf: WsBuf<f32>,
    beta_buf: WsBuf<f32>,
    post_conv_f16: WsBuf<f16>,
    post_conv_bf16: WsBuf<bf16>,
    post_conv_f32: WsBuf<f32>,
    state_o: WsBuf<f32>,
    out_o: WsBuf<f32>,
}

impl PrefillWs {
    fn bytes(&self) -> usize {
        self.qe.bytes()
            + self.ke.bytes()
            + self.vv.bytes()
            + self.g_buf.bytes()
            + self.beta_buf.bytes()
            + self.post_conv_f16.bytes()
            + self.post_conv_bf16.bytes()
            + self.post_conv_f32.bytes()
            + self.state_o.bytes()
            + self.out_o.bytes()
    }
}

static WS: OnceLock<Mutex<HashMap<usize, PrefillWs>>> = OnceLock::new();

/// Отдать пулу скретчи linear-prefill'а. Вызывается при выгрузке модели.
pub fn clear_linear_prefill_ws() -> usize {
    let Some(cache) = WS.get() else { return 0 };
    let mut g = cache.lock();
    let freed = g.values().map(PrefillWs::bytes).sum();
    g.clear();
    freed
}

/// Полная device-резидентная prefill-цепочка (см. модуль-док). Вход `qkv` /
/// `conv_w` / `conv_state` — у одного `compute_dtype` (F16/BF16/F32);
/// `a`/`b` — всегда F16; `dt_bias`/`a_log` — F32; `ssm_state` и `out` — F32.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_chunk_prefill_u8_dev(
    conv_kernels: &CausalConv1dKernels,
    prep_kernels: &LinearAttnRawKernels,
    cfk: &ChunkFlaKernels,
    csk: &ChunkScanKernels,
    stream: &Arc<CudaStream>,
    qkv: (&CudaSlice<u8>, usize),
    conv_w: (&CudaSlice<u8>, usize),
    a: (&CudaSlice<u8>, usize),
    b_in: (&CudaSlice<u8>, usize),
    dt_bias: (&CudaSlice<u8>, usize),
    a_log: (&CudaSlice<u8>, usize),
    conv_state: &mut CudaSlice<u8>,
    conv_state_off: usize,
    ssm_state: &mut CudaSlice<u8>,
    ssm_state_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    compute_dtype: DType,
    num_k: u32,
    num_v: u32,
    hk: u32,
    hv: u32,
    conv_kernel: u32,
    t_in: u32,
    t_pad: u32,
    chunk_size: u32,
    q_scale: f32,
    silu: bool,
) -> Result<()> {
    if num_k == 0 || num_v % num_k != 0 {
        return Err(SynaptixError::Cuda(format!(
            "linear_prefill: num_v {num_v} must be multiple of num_k {num_k}"
        )));
    }
    if t_in == 0 {
        return Ok(());
    }
    if t_pad < t_in {
        return Err(SynaptixError::Cuda(format!(
            "linear_prefill: t_pad={t_pad} < t_in={t_in}"
        )));
    }
    if chunk_size == 0 || t_pad % chunk_size != 0 {
        return Err(SynaptixError::Cuda(format!(
            "linear_prefill: t_pad={t_pad} not divisible by chunk_size={chunk_size}"
        )));
    }
    let n_rep = num_v / num_k;
    let key_dim = num_k * hk;
    let value_dim = num_v * hv;
    let conv_dim = key_dim * 2 + value_dim;
    let t = t_in as usize;
    let tp = t_pad as usize;
    let cd = conv_dim as usize;
    let state_rows = (conv_kernel as usize).saturating_sub(1);
    let conv_w_len = cd * conv_kernel as usize;
    let aerr = |what: &str, e: cudarc::driver::DriverError| {
        SynaptixError::Cuda(format!("linear_prefill alloc {what}: {e:?}"))
    };
    let cerr = |what: &str| SynaptixError::Cuda(format!("linear_prefill transmute {what}"));

    // F32 промежуточные (size = t_pad для kratности chunk_size; для t∈[t_in, t_pad)
    // позиции остаются нулями, см. prep_scatter t_out параметр).
    let ord = stream.context().ordinal();
    let cache = WS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock();
    let ws: &mut PrefillWs = guard.entry(ord).or_default();
    let qe = ws
        .qe
        .fit_zeros(stream, num_v as usize * tp * hk as usize)
        .map_err(|e| aerr("qe", e))?;
    let ke = ws
        .ke
        .fit_zeros(stream, num_v as usize * tp * hk as usize)
        .map_err(|e| aerr("ke", e))?;
    let vv = ws
        .vv
        .fit_zeros(stream, num_v as usize * tp * hv as usize)
        .map_err(|e| aerr("vv", e))?;
    let g_buf = ws.g_buf.fit_zeros(stream, num_v as usize * tp).map_err(|e| aerr("g", e))?;
    let beta_buf = ws
        .beta_buf
        .fit_zeros(stream, num_v as usize * tp)
        .map_err(|e| aerr("beta", e))?;

    // F16 для a/b/dt_bias/a_log — типизированные view'ы.
    let a_v = unsafe {
        a.0.slice(a.1..a.1 + t * num_v as usize * tm::<f16>())
            .transmute::<f16>(t * num_v as usize)
            .ok_or_else(|| cerr("a"))?
    };
    let b_v = unsafe {
        b_in.0
            .slice(b_in.1..b_in.1 + t * num_v as usize * tm::<f16>())
            .transmute::<f16>(t * num_v as usize)
            .ok_or_else(|| cerr("b"))?
    };
    let dt_v = unsafe {
        dt_bias
            .0
            .slice(dt_bias.1..dt_bias.1 + num_v as usize * tm::<f32>())
            .transmute::<f32>(num_v as usize)
            .ok_or_else(|| cerr("dt_bias"))?
    };
    let al_v = unsafe {
        a_log
            .0
            .slice(a_log.1..a_log.1 + num_v as usize * tm::<f32>())
            .transmute::<f32>(num_v as usize)
            .ok_or_else(|| cerr("a_log"))?
    };

    // 1+2. Conv1d-chunk → post_conv; prep_scatter → qe/ke/vv/g/β.
    match compute_dtype {
        DType::F16 => {
            let qkv_v = unsafe {
                qkv.0
                    .slice(qkv.1..qkv.1 + t * cd * tm::<f16>())
                    .transmute::<f16>(t * cd)
                    .ok_or_else(|| cerr("qkv f16"))?
            };
            let w_v = unsafe {
                conv_w
                    .0
                    .slice(conv_w.1..conv_w.1 + conv_w_len * tm::<f16>())
                    .transmute::<f16>(conv_w_len)
                    .ok_or_else(|| cerr("conv_w f16"))?
            };
            let mut cs_s = conv_state
                .slice_mut(conv_state_off..conv_state_off + state_rows * cd * tm::<f16>());
            let mut cs_v = unsafe {
                cs_s.transmute_mut::<f16>(state_rows * cd)
                    .ok_or_else(|| cerr("conv_state f16"))?
            };
            let post_conv = ws
                .post_conv_f16
                .fit_zeros(stream, t * cd)
                .map_err(|e| aerr("post_conv f16", e))?;
            {
                let mut post_conv_v = post_conv.as_view_mut();
                causal_conv1d_chunk_f16(
                    conv_kernels,
                    stream,
                    &qkv_v,
                    &mut cs_v,
                    &w_v,
                    &mut post_conv_v,
                    t_in,
                    conv_dim,
                    conv_kernel,
                    silu,
                )?;
            }
            let post_conv_ro = post_conv.as_view();
            prep_kernels.linear_attn_prep_scatter_f16(
                stream,
                &b_v,
                &a_v,
                &dt_v,
                &al_v,
                &mut *beta_buf,
                &mut *g_buf,
                &post_conv_ro,
                &mut *qe,
                &mut *ke,
                &mut *vv,
                t_in,
                t_pad,
                num_v,
                num_k,
                n_rep,
                hk,
                hv,
            )?;
        }
        DType::BF16 => {
            let qkv_v = unsafe {
                qkv.0
                    .slice(qkv.1..qkv.1 + t * cd * tm::<bf16>())
                    .transmute::<bf16>(t * cd)
                    .ok_or_else(|| cerr("qkv bf16"))?
            };
            let w_v = unsafe {
                conv_w
                    .0
                    .slice(conv_w.1..conv_w.1 + conv_w_len * tm::<bf16>())
                    .transmute::<bf16>(conv_w_len)
                    .ok_or_else(|| cerr("conv_w bf16"))?
            };
            let mut cs_s = conv_state
                .slice_mut(conv_state_off..conv_state_off + state_rows * cd * tm::<bf16>());
            let mut cs_v = unsafe {
                cs_s.transmute_mut::<bf16>(state_rows * cd)
                    .ok_or_else(|| cerr("conv_state bf16"))?
            };
            let post_conv = ws
                .post_conv_bf16
                .fit_zeros(stream, t * cd)
                .map_err(|e| aerr("post_conv bf16", e))?;
            {
                let mut post_conv_v = post_conv.as_view_mut();
                causal_conv1d_chunk_bf16(
                    conv_kernels,
                    stream,
                    &qkv_v,
                    &mut cs_v,
                    &w_v,
                    &mut post_conv_v,
                    t_in,
                    conv_dim,
                    conv_kernel,
                    silu,
                )?;
            }
            let post_conv_ro = post_conv.as_view();
            prep_kernels.linear_attn_prep_scatter_bf16(
                stream,
                &b_v,
                &a_v,
                &dt_v,
                &al_v,
                &mut *beta_buf,
                &mut *g_buf,
                &post_conv_ro,
                &mut *qe,
                &mut *ke,
                &mut *vv,
                t_in,
                t_pad,
                num_v,
                num_k,
                n_rep,
                hk,
                hv,
            )?;
        }
        DType::F32 => {
            let qkv_v = unsafe {
                qkv.0
                    .slice(qkv.1..qkv.1 + t * cd * tm::<f32>())
                    .transmute::<f32>(t * cd)
                    .ok_or_else(|| cerr("qkv f32"))?
            };
            let w_v = unsafe {
                conv_w
                    .0
                    .slice(conv_w.1..conv_w.1 + conv_w_len * tm::<f32>())
                    .transmute::<f32>(conv_w_len)
                    .ok_or_else(|| cerr("conv_w f32"))?
            };
            let mut cs_s = conv_state
                .slice_mut(conv_state_off..conv_state_off + state_rows * cd * tm::<f32>());
            let mut cs_v = unsafe {
                cs_s.transmute_mut::<f32>(state_rows * cd)
                    .ok_or_else(|| cerr("conv_state f32"))?
            };
            let post_conv = ws
                .post_conv_f32
                .fit_zeros(stream, t * cd)
                .map_err(|e| aerr("post_conv f32", e))?;
            {
                let mut post_conv_v = post_conv.as_view_mut();
                causal_conv1d_chunk_f32(
                    conv_kernels,
                    stream,
                    &qkv_v,
                    &mut cs_v,
                    &w_v,
                    &mut post_conv_v,
                    t_in,
                    conv_dim,
                    conv_kernel,
                    silu,
                )?;
            }
            let post_conv_ro = post_conv.as_view();
            prep_kernels.linear_attn_prep_scatter_f32(
                stream,
                &b_v,
                &a_v,
                &dt_v,
                &al_v,
                &mut *beta_buf,
                &mut *g_buf,
                &post_conv_ro,
                &mut *qe,
                &mut *ke,
                &mut *vv,
                t_in,
                t_pad,
                num_v,
                num_k,
                n_rep,
                hk,
                hv,
            )?;
        }
        other => {
            return Err(SynaptixError::Cuda(format!(
                "linear_prefill: compute_dtype {other:?} not supported (F32/F16/BF16)"
            )))
        }
    }

    // 3. Chunked gated-delta-rule prefill — owned F32 buffers (как
    // `gated_delta_rule_prefill` в cuda_backend.rs), затем copy в Storage.
    // scan работает на t_pad → out layout = `[num_v, t_pad, hv]` (caller
    // обязан аллоцировать out с этим size и сам narrow'нуть до t_in).
    let n_state = (num_v * hk * hv) as usize;
    let n_out = (num_v * t_pad * hv) as usize;
    let state_o = ws.state_o.fit_zeros(stream, n_state).map_err(|e| aerr("ssm_state owned", e))?;
    let out_o = ws.out_o.fit_zeros(stream, n_out).map_err(|e| aerr("out owned", e))?;
    {
        // ssm_state Storage → state_o
        let ss_s_v = ssm_state.slice(ssm_state_off..ssm_state_off + n_state * tm::<f32>());
        let ss_f = unsafe {
            ss_s_v
                .transmute::<f32>(n_state)
                .ok_or_else(|| cerr("ssm_state in"))?
        };
        stream
            .memcpy_dtod(&ss_f, &mut *state_o)
            .map_err(|e| SynaptixError::Cuda(format!("linear_prefill memcpy ss in: {e:?}")))?;
    }
    chunk_gated_delta_rule(
        cfk,
        csk,
        stream,
        &*qe,
        &*ke,
        &*vv,
        &*g_buf,
        &*beta_buf,
        &mut *state_o,
        &mut *out_o,
        q_scale,
        num_v,
        t_pad,
        hk,
        hv,
        chunk_size,
    )?;
    {
        let mut ss_s = ssm_state.slice_mut(ssm_state_off..ssm_state_off + n_state * tm::<f32>());
        let mut ss_f = unsafe {
            ss_s.transmute_mut::<f32>(n_state)
                .ok_or_else(|| cerr("ssm_state out"))?
        };
        stream
            .memcpy_dtod(&*state_o, &mut ss_f)
            .map_err(|e| SynaptixError::Cuda(format!("linear_prefill memcpy ss out: {e:?}")))?;
    }
    {
        let mut out_s = out.slice_mut(out_off..out_off + n_out * tm::<f32>());
        let mut out_f = unsafe {
            out_s
                .transmute_mut::<f32>(n_out)
                .ok_or_else(|| cerr("out out"))?
        };
        stream
            .memcpy_dtod(&*out_o, &mut out_f)
            .map_err(|e| SynaptixError::Cuda(format!("linear_prefill memcpy out: {e:?}")))?;
    }
    Ok(())
}

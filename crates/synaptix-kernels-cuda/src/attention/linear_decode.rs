//! Оркестратор device-резидентного decode-шага (T=1) GatedDeltaNet linear-attn
//! для CUDA-graph. Связывает три уже валидированных bit-exact ядра в одну
//! capture-safe последовательность (нет host round-trip, переменной launch-config
//! и записи в host-память):
//!
//!   1. `causal_conv1d_update` (state in-place, FIFO) + SiLU → post_conv (F16);
//!   2. `linear_attn_prep_fused` → β, g (F32) и q/k/v (F32, GQA repeat Q/K);
//!   3. `gated_delta_rule_step_fused_rms_norm` (ssm-state in-place) + RmsNormGated
//!      (× silu(z) × weight) → out (F16).
//!
//! Все промежуточные тензоры — owned scratch (alloc из mempool, capture-safe);
//! `conv_state`/`ssm_state`/`out` приходят device-резидентными u8-views из
//! persistent-буферов (стабильные указатели для graph replay). Семантика ==
//! `synaptix_ops::...::{causal_conv1d_stateful (s=1) + silu, gated_delta_decay_beta,
//! gated_delta_net_recurrent}` + RmsNormGated с точностью до F16-compute.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use half::f16;
use synaptix_core::error::{Result, SynaptixError};

use crate::attention::linear_attn_raw::LinearAttnRawKernels;
use crate::conv::causal_conv1d::CausalConv1dKernels;
use crate::ssm::gated_delta_rule::GatedDeltaRuleKernels;
use crate::wsalloc::WsAlloc;

const CONV_BLOCK: u32 = 128;

fn tm<T>() -> usize {
    std::mem::size_of::<T>()
}

/// Один decode-шаг linear-attn слоя. Входы — `(slice, byte_offset)` поверх
/// untyped u8-storage; dtype фиксирован: GEMM-выходы (`qkv`/`a`/`b`/`z`) и веса
/// (`conv_w`/`norm_w`) — F16; `dt_bias`/`a_log` и `ssm_state` — F32; `conv_state`
/// — F16. `out` — F16 `[value_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn linear_attn_decode_step_u8_dev(
    conv_kernels: &CausalConv1dKernels,
    prep_kernels: &LinearAttnRawKernels,
    gdr_kernels: &GatedDeltaRuleKernels,
    stream: &Arc<CudaStream>,
    qkv: (&CudaSlice<u8>, usize),
    conv_w: (&CudaSlice<u8>, usize),
    a: (&CudaSlice<u8>, usize),
    b_in: (&CudaSlice<u8>, usize),
    dt_bias: (&CudaSlice<u8>, usize),
    a_log: (&CudaSlice<u8>, usize),
    z: (&CudaSlice<u8>, usize),
    norm_w: (&CudaSlice<u8>, usize),
    conv_state: &mut CudaSlice<u8>,
    conv_state_off: usize,
    ssm_state: &mut CudaSlice<u8>,
    ssm_state_off: usize,
    out: &mut CudaSlice<u8>,
    out_off: usize,
    num_k: u32,
    num_v: u32,
    dk: u32,
    dv: u32,
    conv_kernel: u32,
    q_scale: f32,
    eps: f32,
) -> Result<()> {
    if num_k == 0 || num_v % num_k != 0 {
        return Err(SynaptixError::Cuda(format!(
            "linear_decode: num_v {num_v} must be multiple of num_k {num_k}"
        )));
    }
    let (nv, hk, hv) = (num_v as usize, dk as usize, dv as usize);
    let key_dim = num_k as usize * hk;
    let value_dim = nv * hv;
    let conv_dim = key_dim * 2 + value_dim;
    let conv_w_len = conv_dim * conv_kernel as usize;
    let state_rows = (conv_kernel as usize).saturating_sub(1);
    let n_rep = num_v / num_k;

    let cerr = |what: &str| SynaptixError::Cuda(format!("linear_decode transmute {what}"));
    let lerr = |what: &str, e: cudarc::driver::DriverError| {
        SynaptixError::Cuda(format!("launch {what}: {e:?}"))
    };
    let aerr = |what: &str, e: cudarc::driver::DriverError| {
        SynaptixError::Cuda(format!("linear_decode alloc {what}: {e:?}"))
    };

    let mut post_conv = stream
        .ws_alloc_zeros::<f16>(conv_dim)
        .map_err(|e| aerr("post_conv", e))?;
    let mut beta = stream.ws_alloc_zeros::<f32>(nv).map_err(|e| aerr("beta", e))?;
    let mut g = stream.ws_alloc_zeros::<f32>(nv).map_err(|e| aerr("g", e))?;
    let mut q = stream
        .ws_alloc_zeros::<f32>(nv * hk)
        .map_err(|e| aerr("q", e))?;
    let mut k = stream
        .ws_alloc_zeros::<f32>(nv * hk)
        .map_err(|e| aerr("k", e))?;
    let mut v = stream
        .ws_alloc_zeros::<f32>(nv * hv)
        .map_err(|e| aerr("v", e))?;

    // 1. causal conv1d update (in-place state) + SiLU.
    {
        let x_v = unsafe {
            qkv.0
                .slice(qkv.1..qkv.1 + conv_dim * tm::<f16>())
                .transmute::<f16>(conv_dim)
                .ok_or_else(|| cerr("qkv"))?
        };
        let w_v = unsafe {
            conv_w
                .0
                .slice(conv_w.1..conv_w.1 + conv_w_len * tm::<f16>())
                .transmute::<f16>(conv_w_len)
                .ok_or_else(|| cerr("conv_w"))?
        };
        let mut cs_s = conv_state
            .slice_mut(conv_state_off..conv_state_off + state_rows * conv_dim * tm::<f16>());
        let mut cs_v = unsafe {
            cs_s.transmute_mut::<f16>(state_rows * conv_dim)
                .ok_or_else(|| cerr("conv_state"))?
        };
        let cfg = LaunchConfig {
            grid_dim: ((conv_dim as u32).div_ceil(CONV_BLOCK), 1, 1),
            block_dim: (CONV_BLOCK, 1, 1),
            shared_mem_bytes: 0,
        };
        let (cd_i, k_i, silu_i) = (conv_dim as i32, conv_kernel as i32, 1i32);
        let mut bld = stream.launch_builder(conv_kernels.update_f16_fn());
        bld.arg(&x_v)
            .arg(&mut cs_v)
            .arg(&w_v)
            .arg(&mut post_conv)
            .arg(&cd_i)
            .arg(&k_i)
            .arg(&silu_i);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| lerr("causal_conv1d_update", e))?;
        }
    }

    // 2. prep_fused: β = sigmoid(b); g = softplus(a+dt_bias)·(−exp(a_log));
    //    q/k = repeat_interleave(post_conv[Q|K], n_rep); v = post_conv[V].
    {
        let a_v = unsafe {
            a.0.slice(a.1..a.1 + nv * tm::<f16>())
                .transmute::<f16>(nv)
                .ok_or_else(|| cerr("a"))?
        };
        let b_v = unsafe {
            b_in.0
                .slice(b_in.1..b_in.1 + nv * tm::<f16>())
                .transmute::<f16>(nv)
                .ok_or_else(|| cerr("b"))?
        };
        let dt_v = unsafe {
            dt_bias
                .0
                .slice(dt_bias.1..dt_bias.1 + nv * tm::<f32>())
                .transmute::<f32>(nv)
                .ok_or_else(|| cerr("dt_bias"))?
        };
        let al_v = unsafe {
            a_log
                .0
                .slice(a_log.1..a_log.1 + nv * tm::<f32>())
                .transmute::<f32>(nv)
                .ok_or_else(|| cerr("a_log"))?
        };
        let block = hk.max(hv) as u32;
        let cfg = LaunchConfig {
            grid_dim: (num_v, 1, 4),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let key_dim_u = key_dim as u32;
        let mut bld = stream.launch_builder(prep_kernels.prep_fused_fn());
        bld.arg(&b_v)
            .arg(&a_v)
            .arg(&dt_v)
            .arg(&al_v)
            .arg(&mut beta)
            .arg(&mut g)
            .arg(&post_conv)
            .arg(&mut q)
            .arg(&mut k)
            .arg(&mut v)
            .arg(&num_v)
            .arg(&n_rep)
            .arg(&dk)
            .arg(&dv)
            .arg(&key_dim_u);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| lerr("linear_attn_prep_fused", e))?;
        }
    }

    // 3. gated_delta_rule_step (ssm-state in-place) + RmsNormGated → out.
    {
        let gate_v = unsafe {
            z.0.slice(z.1..z.1 + value_dim * tm::<f16>())
                .transmute::<f16>(value_dim)
                .ok_or_else(|| cerr("z"))?
        };
        let nw_v = unsafe {
            norm_w
                .0
                .slice(norm_w.1..norm_w.1 + hv * tm::<f16>())
                .transmute::<f16>(hv)
                .ok_or_else(|| cerr("norm_w"))?
        };
        let mut ss_s =
            ssm_state.slice_mut(ssm_state_off..ssm_state_off + nv * hk * hv * tm::<f32>());
        let mut ss_v = unsafe {
            ss_s.transmute_mut::<f32>(nv * hk * hv)
                .ok_or_else(|| cerr("ssm_state"))?
        };
        let mut out_s = out.slice_mut(out_off..out_off + value_dim * tm::<f16>());
        let mut out_v = unsafe {
            out_s
                .transmute_mut::<f16>(value_dim)
                .ok_or_else(|| cerr("out"))?
        };
        let shared = ((3 * dk + dv + 4) as usize * tm::<f32>()) as u32;
        let cfg = LaunchConfig {
            grid_dim: (1, num_v, 1),
            block_dim: (dk, 1, 1),
            shared_mem_bytes: shared,
        };
        let b_u = 1u32;
        let mut bld = stream.launch_builder(gdr_kernels.step_fused_rms_fn());
        bld.arg(&q)
            .arg(&k)
            .arg(&v)
            .arg(&g)
            .arg(&beta)
            .arg(&mut ss_v)
            .arg(&gate_v)
            .arg(&nw_v)
            .arg(&mut out_v)
            .arg(&q_scale)
            .arg(&eps)
            .arg(&b_u)
            .arg(&num_v)
            .arg(&dk)
            .arg(&dv);
        unsafe {
            bld.launch(cfg)
                .map_err(|e| lerr("gated_delta_rule_step_fused_rms_norm", e))?;
        }
    }

    Ok(())
}

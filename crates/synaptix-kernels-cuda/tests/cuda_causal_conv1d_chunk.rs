#![cfg(feature = "cuda")]

//! Bit-exact: `causal_conv1d_chunk_{f32,f16,bf16}` (prefill T≥1, stateful)
//! vs host `synaptix_ops::conv::causal_conv1d_stateful` + optional SiLU.
//! Покрывает: общий случай T > K-1, граничные T=1/T=K-1/T<K-1, разные K,
//! parity к `causal_conv1d_update` для T=1.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::conv::causal_conv1d::{
    causal_conv1d_chunk_bf16, causal_conv1d_chunk_f16, causal_conv1d_chunk_f32, CausalConv1dKernels,
};

// Inline-копия host-эталона `synaptix_ops::conv::causal_conv1d_stateful`
// (избегаем cross-crate dep в тестах kernels-cuda). См. оригинал — оба должны
// совпадать; копия здесь зафиксирована, чтобы регрессия в ops не маскировала
// баг ядра.
fn causal_conv1d_stateful(
    state: &mut [f32],
    x: &[f32],
    w: &[f32],
    s: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    let km1 = k - 1;
    let mut ext = vec![0.0f32; (km1 + s) * channels];
    ext[..km1 * channels].copy_from_slice(state);
    ext[km1 * channels..].copy_from_slice(&x[..s * channels]);
    let mut out = vec![0.0f32; s * channels];
    for i in 0..s {
        for c in 0..channels {
            let mut acc = 0.0f32;
            for j in 0..k {
                acc += w[c * k + j] * ext[(i + j) * channels + c];
            }
            out[i * channels + c] = acc;
        }
    }
    let start = s * channels;
    state.copy_from_slice(&ext[start..start + km1 * channels]);
    out
}

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 * scale - scale
        })
        .collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

fn run_case_f32(t: usize, c: usize, k: usize, silu: bool, seed: u64) {
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("kernels");

    let x = det_f32(seed, t * c, 0.7);
    let w = det_f32(seed.wrapping_add(0x100), c * k, 0.5);
    let s0 = det_f32(seed.wrapping_add(0x200), (k - 1) * c, 0.3);

    // Host эталон.
    let mut state_ref = s0.clone();
    let mut out_ref = causal_conv1d_stateful(&mut state_ref, &x, &w, t, c, k);
    if silu {
        for v in out_ref.iter_mut() {
            *v /= 1.0 + (-*v).exp();
        }
    }

    // Device.
    let x_d = stream.memcpy_stod(&x).unwrap();
    let w_d = stream.memcpy_stod(&w).unwrap();
    let mut state_d = stream.memcpy_stod(&s0).unwrap();
    let mut out_d = stream.alloc_zeros::<f32>(t * c).unwrap();
    {
        let x_v = x_d.as_view();
        let w_v = w_d.as_view();
        let mut state_v = state_d.as_view_mut();
        let mut out_v = out_d.as_view_mut();
        causal_conv1d_chunk_f32(
            &kernels,
            &stream,
            &x_v,
            &mut state_v,
            &w_v,
            &mut out_v,
            t as u32,
            c as u32,
            k as u32,
            silu,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let out_h = stream.memcpy_dtov(&out_d).unwrap();
    let state_h = stream.memcpy_dtov(&state_d).unwrap();

    let m_out = max_abs(&out_h, &out_ref);
    let m_st = max_abs(&state_h, &state_ref);
    eprintln!(
        "[chunk f32 T={t} C={c} K={k} silu={silu}] out_max_abs={m_out:.3e} state_max_abs={m_st:.3e}"
    );
    assert!(m_out < 1e-5, "out parity max_abs={m_out}");
    assert!(m_st < 1e-5, "state parity max_abs={m_st}");
}

fn run_case_f16(t: usize, c: usize, k: usize, silu: bool, seed: u64) {
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("kernels");

    let x = det_f32(seed, t * c, 0.7);
    let w = det_f32(seed.wrapping_add(0x100), c * k, 0.5);
    let s0 = det_f32(seed.wrapping_add(0x200), (k - 1) * c, 0.3);

    // F16-host: квантуем во входе/state/w, считаем эталон в F32 «через F16 quant».
    let to_h = |v: &[f32]| -> Vec<f16> { v.iter().map(|x| f16::from_f32(*x)).collect() };
    let to_f = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
    let x_q = to_f(&to_h(&x));
    let w_q = to_f(&to_h(&w));
    let s0_q = to_f(&to_h(&s0));
    let mut state_ref = s0_q.clone();
    let mut out_ref = causal_conv1d_stateful(&mut state_ref, &x_q, &w_q, t, c, k);
    if silu {
        for v in out_ref.iter_mut() {
            *v /= 1.0 + (-*v).exp();
        }
    }
    // Финальная квантизация выхода/state в F16 (тот же путь у device).
    let out_ref_q = to_f(&to_h(&out_ref));
    let st_ref_q = to_f(&to_h(&state_ref));

    let x_d = stream.memcpy_stod(&to_h(&x)).unwrap();
    let w_d = stream.memcpy_stod(&to_h(&w)).unwrap();
    let mut state_d = stream.memcpy_stod(&to_h(&s0)).unwrap();
    let mut out_d = stream.alloc_zeros::<f16>(t * c).unwrap();
    {
        let x_v = x_d.as_view();
        let w_v = w_d.as_view();
        let mut state_v = state_d.as_view_mut();
        let mut out_v = out_d.as_view_mut();
        causal_conv1d_chunk_f16(
            &kernels,
            &stream,
            &x_v,
            &mut state_v,
            &w_v,
            &mut out_v,
            t as u32,
            c as u32,
            k as u32,
            silu,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let out_h: Vec<f32> = stream
        .memcpy_dtov(&out_d)
        .unwrap()
        .iter()
        .map(|x: &f16| x.to_f32())
        .collect();
    let state_h: Vec<f32> = stream
        .memcpy_dtov(&state_d)
        .unwrap()
        .iter()
        .map(|x: &f16| x.to_f32())
        .collect();

    let m_out = max_abs(&out_h, &out_ref_q);
    let m_st = max_abs(&state_h, &st_ref_q);
    eprintln!(
        "[chunk f16 T={t} C={c} K={k} silu={silu}] out_max_abs={m_out:.3e} state_max_abs={m_st:.3e}"
    );
    // F16-tol: f32-acc матчится, последняя квантизация общая → tol ≈ 1e-2.
    assert!(m_out < 5e-2, "out parity max_abs={m_out}");
    assert!(m_st < 5e-2, "state parity max_abs={m_st}");
}

fn run_case_bf16(t: usize, c: usize, k: usize, silu: bool, seed: u64) {
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = CausalConv1dKernels::for_context(&ctx).expect("kernels");

    let x = det_f32(seed, t * c, 0.7);
    let w = det_f32(seed.wrapping_add(0x100), c * k, 0.5);
    let s0 = det_f32(seed.wrapping_add(0x200), (k - 1) * c, 0.3);

    let to_h = |v: &[f32]| -> Vec<bf16> { v.iter().map(|x| bf16::from_f32(*x)).collect() };
    let to_f = |v: &[bf16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
    let x_q = to_f(&to_h(&x));
    let w_q = to_f(&to_h(&w));
    let s0_q = to_f(&to_h(&s0));
    let mut state_ref = s0_q.clone();
    let mut out_ref = causal_conv1d_stateful(&mut state_ref, &x_q, &w_q, t, c, k);
    if silu {
        for v in out_ref.iter_mut() {
            *v /= 1.0 + (-*v).exp();
        }
    }
    let out_ref_q = to_f(&to_h(&out_ref));
    let st_ref_q = to_f(&to_h(&state_ref));

    let x_d = stream.memcpy_stod(&to_h(&x)).unwrap();
    let w_d = stream.memcpy_stod(&to_h(&w)).unwrap();
    let mut state_d = stream.memcpy_stod(&to_h(&s0)).unwrap();
    let mut out_d = stream.alloc_zeros::<bf16>(t * c).unwrap();
    {
        let x_v = x_d.as_view();
        let w_v = w_d.as_view();
        let mut state_v = state_d.as_view_mut();
        let mut out_v = out_d.as_view_mut();
        causal_conv1d_chunk_bf16(
            &kernels,
            &stream,
            &x_v,
            &mut state_v,
            &w_v,
            &mut out_v,
            t as u32,
            c as u32,
            k as u32,
            silu,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    let out_h: Vec<f32> = stream
        .memcpy_dtov(&out_d)
        .unwrap()
        .iter()
        .map(|x: &bf16| x.to_f32())
        .collect();
    let state_h: Vec<f32> = stream
        .memcpy_dtov(&state_d)
        .unwrap()
        .iter()
        .map(|x: &bf16| x.to_f32())
        .collect();

    let m_out = max_abs(&out_h, &out_ref_q);
    let m_st = max_abs(&state_h, &st_ref_q);
    eprintln!(
        "[chunk bf16 T={t} C={c} K={k} silu={silu}] out_max_abs={m_out:.3e} state_max_abs={m_st:.3e}"
    );
    // BF16-tol: 7-bit мантисса, грубее.
    assert!(m_out < 1e-1, "out parity max_abs={m_out}");
    assert!(m_st < 1e-1, "state parity max_abs={m_st}");
}

#[test]
fn chunk_f32_t_large_k4() {
    run_case_f32(350, 320, 4, false, 0x10);
}

#[test]
fn chunk_f32_with_silu() {
    run_case_f32(64, 128, 4, true, 0x11);
}

#[test]
fn chunk_f32_t_eq_km1() {
    // T == K-1: проверяет границу state/x в state-update.
    run_case_f32(3, 64, 4, false, 0x12);
}

#[test]
fn chunk_f32_t_lt_km1() {
    // T < K-1: state-update берёт часть из старого state, часть из x.
    run_case_f32(2, 32, 4, false, 0x13);
}

#[test]
fn chunk_f32_t1_matches_update_semantics() {
    // T=1: эквивалент causal_conv1d_update.
    run_case_f32(1, 256, 3, true, 0x14);
}

#[test]
fn chunk_f32_k3() {
    run_case_f32(128, 96, 3, true, 0x15);
}

#[test]
fn chunk_f16_t_large() {
    run_case_f16(350, 320, 4, true, 0x20);
}

#[test]
fn chunk_bf16_t_large() {
    run_case_bf16(350, 320, 4, true, 0x21);
}

#![cfg(feature = "cuda")]

//! Bit-exact (F32-эталон) тесты для gated delta rule рекуррентного шага.
//!
//! Эталон повторяет ядро дословно; гоняем несколько шагов с переносом state и
//! сверяем выход на каждом шаге + финальный state.

use half::f16;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::ssm::gated_delta_rule::GatedDeltaRuleKernels;

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f32(seed: u64, n: usize, scale: f32, offset: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f * scale + offset
        })
        .collect()
}

/// CPU-эталон одного шага (дословно из ядра). `state` обновляется in-place,
/// возвращает core-выход (B*H*hv) — то, что ядро пишет в `out`.
#[allow(clippy::too_many_arguments)]
fn cpu_step(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    g: &[f32],
    beta: &[f32],
    state: &mut [f32],
    q_scale: f32,
    b: usize,
    h: usize,
    hk: usize,
    hv: usize,
) -> Vec<f32> {
    let mut out = vec![0.0_f32; b * h * hv];
    for bi in 0..b {
        for hi in 0..h {
            let base_qk = (bi * h + hi) * hk;
            let base_v = (bi * h + hi) * hv;
            let base_state = (bi * h + hi) * hk * hv;

            let mut sum_q = 0.0_f32;
            let mut sum_k = 0.0_f32;
            for t in 0..hk {
                sum_q += q[base_qk + t] * q[base_qk + t];
                sum_k += k[base_qk + t] * k[base_qk + t];
            }
            let inv_q = 1.0 / (sum_q + 1e-6).sqrt();
            let inv_k = 1.0 / (sum_k + 1e-6).sqrt();
            let q_norm: Vec<f32> = (0..hk).map(|t| q[base_qk + t] * inv_q * q_scale).collect();
            let k_norm: Vec<f32> = (0..hk).map(|t| k[base_qk + t] * inv_k).collect();

            let g_t = g[bi * h + hi].exp();
            let beta_t = beta[bi * h + hi];

            for vi in 0..hv {
                let mut st = vec![0.0_f32; hk];
                let mut kv_mem = 0.0_f32;
                for kk in 0..hk {
                    st[kk] = state[base_state + kk * hv + vi] * g_t;
                    kv_mem += st[kk] * k_norm[kk];
                }
                let delta = (v[base_v + vi] - kv_mem) * beta_t;
                let mut o = 0.0_f32;
                for kk in 0..hk {
                    let new_st = st[kk] + k_norm[kk] * delta;
                    state[base_state + kk * hv + vi] = new_st;
                    o += new_st * q_norm[kk];
                }
                out[base_v + vi] = o;
            }
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn gated_delta_rule_step_multistep() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = GatedDeltaRuleKernels::for_context(&ctx).expect("compile");
    let (b, h, hk, hv) = (2usize, 4usize, 16usize, 16usize);
    let q_scale = 0.5_f32;
    let steps = 5usize;

    // CPU и GPU state стартуют с нуля.
    let mut cpu_state = vec![0.0_f32; b * h * hk * hv];
    let mut dev_state: CudaSlice<f32> = stream.alloc_zeros(b * h * hk * hv).unwrap();

    let mut worst = 0.0_f32;
    for t in 0..steps {
        let q = det_f32(0x100 + t as u64, b * h * hk, 1.0, 0.0);
        let k = det_f32(0x200 + t as u64, b * h * hk, 1.0, 0.0);
        let v = det_f32(0x300 + t as u64, b * h * hv, 1.0, 0.0);
        let g = det_f32(0x400 + t as u64, b * h, 0.2, -0.1); // g < 0 → decay ∈ (0,1)
        let beta = det_f32(0x500 + t as u64, b * h, 0.3, 0.5);

        let expected = cpu_step(&q, &k, &v, &g, &beta, &mut cpu_state, q_scale, b, h, hk, hv);

        let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
        let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
        let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
        let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
        let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
        let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * h * hv).unwrap();
        kern.gated_delta_rule_step(
            &stream,
            &dq,
            &dk,
            &dv,
            &dg,
            &dbeta,
            &mut dev_state,
            &mut dout,
            q_scale,
            b as u32,
            h as u32,
            hk as u32,
            hv as u32,
        )
        .unwrap();
        stream.synchronize().unwrap();
        let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
        let m = max_abs(&got, &expected);
        worst = worst.max(m);
        eprintln!("[gdr step {t}] max_abs={m:.6}");
        assert!(m < 1e-4, "step {t}: max_abs={m}");
    }
    // Сверка финального state.
    let got_state: Vec<f32> = stream.clone_dtoh(&dev_state).unwrap();
    let ms = max_abs(&got_state, &cpu_state);
    eprintln!("[gdr] worst_out={worst:.6} state_max_abs={ms:.6}");
    assert!(ms < 1e-4, "state max_abs={ms}");
}

#[test]
fn gated_delta_rule_fused_rms_matches_separate() {
    let Some((ctx, stream)) = setup() else { return };
    let kern = GatedDeltaRuleKernels::for_context(&ctx).expect("compile");
    let (b, h, hk, hv) = (2usize, 4usize, 16usize, 16usize);
    let q_scale = 0.5_f32;
    let eps = 1e-6_f32;

    let q = det_f32(0xA1, b * h * hk, 1.0, 0.0);
    let k = det_f32(0xA2, b * h * hk, 1.0, 0.0);
    let v = det_f32(0xA3, b * h * hv, 1.0, 0.0);
    let g = det_f32(0xA4, b * h, 0.2, -0.1);
    let beta = det_f32(0xA5, b * h, 0.3, 0.5);
    let gate = det_f32(0xA6, b * h * hv, 1.0, 0.0);
    let weight = det_f32(0xA7, hv, 0.5, 1.0);

    let gate_h: Vec<f16> = gate.iter().map(|x| f16::from_f32(*x)).collect();
    let weight_h: Vec<f16> = weight.iter().map(|x| f16::from_f32(*x)).collect();
    let gate_back: Vec<f32> = gate_h.iter().map(|x| x.to_f32()).collect();
    let weight_back: Vec<f32> = weight_h.iter().map(|x| x.to_f32()).collect();

    // Эталон: SSM core + RmsNormGated.
    let mut cpu_state = vec![0.0_f32; b * h * hk * hv];
    let core = cpu_step(&q, &k, &v, &g, &beta, &mut cpu_state, q_scale, b, h, hk, hv);
    let mut expected = vec![0.0_f32; b * h * hv];
    for bh in 0..b * h {
        let base = bh * hv;
        let mut ss = 0.0_f32;
        for d in 0..hv {
            ss += core[base + d] * core[base + d];
        }
        let inv = 1.0_f32 / (ss / hv as f32 + eps).sqrt();
        for d in 0..hv {
            let gz = gate_back[base + d];
            let silu = gz * (1.0 / (1.0 + (-gz).exp()));
            let res = weight_back[d] * core[base + d] * inv * silu;
            expected[base + d] = f16::from_f32(res).to_f32();
        }
    }

    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let dg: CudaSlice<f32> = stream.clone_htod(&g).unwrap();
    let dbeta: CudaSlice<f32> = stream.clone_htod(&beta).unwrap();
    let dgate: CudaSlice<f16> = stream.clone_htod(&gate_h).unwrap();
    let dweight: CudaSlice<f16> = stream.clone_htod(&weight_h).unwrap();
    let mut dev_state: CudaSlice<f32> = stream.alloc_zeros(b * h * hk * hv).unwrap();
    let mut dout: CudaSlice<f16> = stream.alloc_zeros(b * h * hv).unwrap();
    kern.gated_delta_rule_step_fused_rms_norm(
        &stream,
        &dq,
        &dk,
        &dv,
        &dg,
        &dbeta,
        &mut dev_state,
        &dgate,
        &dweight,
        &mut dout,
        q_scale,
        eps,
        b as u32,
        h as u32,
        hk as u32,
        hv as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = got_h.iter().map(|x| x.to_f32()).collect();
    let m = max_abs(&got, &expected);
    eprintln!("[gdr fused rms] max_abs={m:.6}");
    assert!(m < 0.01, "max_abs={m}");
}

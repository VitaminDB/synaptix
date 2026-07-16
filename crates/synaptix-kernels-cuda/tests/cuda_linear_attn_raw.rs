#![cfg(feature = "cuda")]

//! Bit-exact (F32-эталон) тесты для prep-ядер linear attention.

use half::f16;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::attention::linear_attn_raw::LinearAttnRawKernels;

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

fn to_f16(v: &[f32]) -> Vec<f16> {
    v.iter().map(|x| f16::from_f32(*x)).collect()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .fold(0.0_f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
fn softplus_neg_exp_g_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LinearAttnRawKernels::for_context(&ctx).expect("compile");
    let num_v = 64usize;
    let a_f = det_f32(0x1111, num_v, 0.5, 0.0);
    let dt_bias = det_f32(0x2222, num_v, 0.3, 0.0);
    let a_log = det_f32(0x3333, num_v, 0.4, -1.0);

    // F16 round-trip для a (kernel читает F16).
    let a_h = to_f16(&a_f);
    let a_back: Vec<f32> = a_h.iter().map(|v| v.to_f32()).collect();
    let expected: Vec<f32> = (0..num_v)
        .map(|i| {
            let dt = a_back[i] + dt_bias[i];
            let softplus = (1.0_f32 + dt.exp()).ln();
            softplus * (-a_log[i].exp())
        })
        .collect();

    let dev_a: CudaSlice<f16> = stream.clone_htod(&a_h).unwrap();
    let dev_dt: CudaSlice<f32> = stream.clone_htod(&dt_bias).unwrap();
    let dev_al: CudaSlice<f32> = stream.clone_htod(&a_log).unwrap();
    let mut dev_g: CudaSlice<f32> = stream.alloc_zeros(num_v).unwrap();
    k.softplus_neg_exp_g(&stream, &dev_a, &dev_dt, &dev_al, &mut dev_g, num_v as u32)
        .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_g).unwrap();
    let m = max_abs(&got, &expected);
    eprintln!("[softplus_neg_exp_g] max_abs={m:.6}");
    assert!(m < 1e-4, "max_abs={m}");
}

#[test]
fn sigmoid_f16_to_f32_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LinearAttnRawKernels::for_context(&ctx).expect("compile");
    let n = 128usize;
    let in_f = det_f32(0x4444, n, 2.0, 0.0);
    let in_h = to_f16(&in_f);
    let in_back: Vec<f32> = in_h.iter().map(|v| v.to_f32()).collect();
    let expected: Vec<f32> = in_back.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();

    let dev_in: CudaSlice<f16> = stream.clone_htod(&in_h).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(n).unwrap();
    k.sigmoid_f16_to_f32(&stream, &dev_in, &mut dev_out, n as u32)
        .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let m = max_abs(&got, &expected);
    eprintln!("[sigmoid_f16_to_f32] max_abs={m:.6}");
    assert!(m < 1e-4, "max_abs={m}");
}

#[test]
fn repeat_interleave_cast_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LinearAttnRawKernels::for_context(&ctx).expect("compile");
    let h_in = 16usize;
    let n_rep = 4usize;
    let dim = 128usize;
    let h_out = h_in * n_rep;
    let in_f = det_f32(0x5555, h_in * dim, 1.0, 0.0);
    let in_h = to_f16(&in_f);
    let in_back: Vec<f32> = in_h.iter().map(|v| v.to_f32()).collect();
    let mut expected = vec![0.0_f32; h_out * dim];
    for ho in 0..h_out {
        let hi = ho / n_rep;
        for d in 0..dim {
            expected[ho * dim + d] = in_back[hi * dim + d];
        }
    }
    let dev_in: CudaSlice<f16> = stream.clone_htod(&in_h).unwrap();
    let mut dev_out: CudaSlice<f32> = stream.alloc_zeros(h_out * dim).unwrap();
    k.repeat_interleave_cast_f16_to_f32(
        &stream,
        &dev_in,
        0,
        &mut dev_out,
        h_in as u32,
        n_rep as u32,
        dim as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dev_out).unwrap();
    let m = max_abs(&got, &expected);
    eprintln!("[repeat_interleave_cast] max_abs={m:.6}");
    assert!(m == 0.0, "max_abs={m}");
}

#[test]
fn rms_norm_gated_matches_ref() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LinearAttnRawKernels::for_context(&ctx).expect("compile");
    let n_rows = 64usize;
    let dim = 128usize;
    let eps = 1e-6_f64;
    let x_f = det_f32(0x6666, n_rows * dim, 1.0, 0.0);
    let gate_f = det_f32(0x7777, n_rows * dim, 1.0, 0.0);
    let weight_f = det_f32(0x8888, dim, 0.5, 1.0);

    let gate_h = to_f16(&gate_f);
    let weight_h = to_f16(&weight_f);
    let gate_back: Vec<f32> = gate_h.iter().map(|v| v.to_f32()).collect();
    let weight_back: Vec<f32> = weight_h.iter().map(|v| v.to_f32()).collect();

    let mut expected = vec![0.0_f32; n_rows * dim];
    for r in 0..n_rows {
        let base = r * dim;
        let mut ss = 0.0_f32;
        for d in 0..dim {
            let v = x_f[base + d];
            ss += v * v;
        }
        let inv = 1.0_f32 / (ss / dim as f32 + eps as f32).sqrt();
        for d in 0..dim {
            let g = gate_back[base + d];
            let silu = g * (1.0 / (1.0 + (-g).exp()));
            let res = weight_back[d] * x_f[base + d] * inv * silu;
            // kernel пишет F16 → эмулируем округление.
            expected[base + d] = f16::from_f32(res).to_f32();
        }
    }

    let dev_x: CudaSlice<f32> = stream.clone_htod(&x_f).unwrap();
    let dev_gate: CudaSlice<f16> = stream.clone_htod(&gate_h).unwrap();
    let dev_w: CudaSlice<f16> = stream.clone_htod(&weight_h).unwrap();
    let mut dev_out: CudaSlice<f16> = stream.alloc_zeros(n_rows * dim).unwrap();
    k.rms_norm_gated_f32_in_f16_out(
        &stream,
        &dev_x,
        &dev_gate,
        &dev_w,
        &mut dev_out,
        eps,
        n_rows as u32,
        dim as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got_h: Vec<f16> = stream.clone_dtoh(&dev_out).unwrap();
    let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
    let m = max_abs(&got, &expected);
    eprintln!("[rms_norm_gated] max_abs={m:.6}");
    assert!(m < 0.01, "max_abs={m}");
}

#[test]
fn prep_fused_matches_individual() {
    let Some((ctx, stream)) = setup() else { return };
    let k = LinearAttnRawKernels::for_context(&ctx).expect("compile");
    let num_k = 16usize;
    let n_rep = 4usize;
    let num_v = num_k * n_rep;
    let hk = 128usize;
    let hv = 128usize;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;

    let b_f = det_f32(0x9001, num_v, 1.5, 0.0);
    let a_f = det_f32(0x9002, num_v, 0.5, 0.0);
    let dt_bias = det_f32(0x9003, num_v, 0.3, 0.0);
    let a_log = det_f32(0x9004, num_v, 0.4, -1.0);
    let post_conv = det_f32(0x9005, conv_dim, 1.0, 0.0);

    let b_h = to_f16(&b_f);
    let a_h = to_f16(&a_f);
    let pc_h = to_f16(&post_conv);
    let b_back: Vec<f32> = b_h.iter().map(|v| v.to_f32()).collect();
    let a_back: Vec<f32> = a_h.iter().map(|v| v.to_f32()).collect();
    let pc_back: Vec<f32> = pc_h.iter().map(|v| v.to_f32()).collect();

    // Эталон.
    let mut beta_exp = vec![0.0_f32; num_v];
    let mut g_exp = vec![0.0_f32; num_v];
    for i in 0..num_v {
        beta_exp[i] = 1.0 / (1.0 + (-b_back[i]).exp());
        let dt = a_back[i] + dt_bias[i];
        g_exp[i] = (1.0_f32 + dt.exp()).ln() * (-a_log[i].exp());
    }
    let mut q_exp = vec![0.0_f32; num_v * hk];
    let mut k_exp = vec![0.0_f32; num_v * hk];
    let mut v_exp = vec![0.0_f32; num_v * hv];
    for ho in 0..num_v {
        let hi = ho / n_rep;
        for d in 0..hk {
            q_exp[ho * hk + d] = pc_back[hi * hk + d];
            k_exp[ho * hk + d] = pc_back[key_dim + hi * hk + d];
        }
        for d in 0..hv {
            v_exp[ho * hv + d] = pc_back[2 * key_dim + ho * hv + d];
        }
    }

    let dev_b: CudaSlice<f16> = stream.clone_htod(&b_h).unwrap();
    let dev_a: CudaSlice<f16> = stream.clone_htod(&a_h).unwrap();
    let dev_dt: CudaSlice<f32> = stream.clone_htod(&dt_bias).unwrap();
    let dev_al: CudaSlice<f32> = stream.clone_htod(&a_log).unwrap();
    let dev_pc: CudaSlice<f16> = stream.clone_htod(&pc_h).unwrap();
    let mut dev_beta: CudaSlice<f32> = stream.alloc_zeros(num_v).unwrap();
    let mut dev_g: CudaSlice<f32> = stream.alloc_zeros(num_v).unwrap();
    let mut dev_q: CudaSlice<f32> = stream.alloc_zeros(num_v * hk).unwrap();
    let mut dev_k: CudaSlice<f32> = stream.alloc_zeros(num_v * hk).unwrap();
    let mut dev_v: CudaSlice<f32> = stream.alloc_zeros(num_v * hv).unwrap();
    k.linear_attn_prep_fused(
        &stream,
        &dev_b,
        &dev_a,
        &dev_dt,
        &dev_al,
        &mut dev_beta,
        &mut dev_g,
        &dev_pc,
        &mut dev_q,
        &mut dev_k,
        &mut dev_v,
        num_v as u32,
        num_k as u32,
        n_rep as u32,
        hk as u32,
        hv as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let beta_got: Vec<f32> = stream.clone_dtoh(&dev_beta).unwrap();
    let g_got: Vec<f32> = stream.clone_dtoh(&dev_g).unwrap();
    let q_got: Vec<f32> = stream.clone_dtoh(&dev_q).unwrap();
    let k_got: Vec<f32> = stream.clone_dtoh(&dev_k).unwrap();
    let v_got: Vec<f32> = stream.clone_dtoh(&dev_v).unwrap();

    let mb = max_abs(&beta_got, &beta_exp);
    let mg = max_abs(&g_got, &g_exp);
    let mq = max_abs(&q_got, &q_exp);
    let mk = max_abs(&k_got, &k_exp);
    let mv = max_abs(&v_got, &v_exp);
    eprintln!("[prep_fused] beta={mb:.6} g={mg:.6} q={mq:.6} k={mk:.6} v={mv:.6}");
    assert!(mb < 1e-4 && mg < 1e-4, "beta={mb} g={mg}");
    assert!(mq == 0.0 && mk == 0.0 && mv == 0.0, "q={mq} k={mk} v={mv}");
}

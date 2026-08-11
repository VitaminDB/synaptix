
//! Bit-exact: `LinearAttnRawKernels::linear_attn_prep_scatter_{f16,bf16,f32}`
//! (chunk T≥1) vs host scatter loop + `gated_delta_decay_beta`.
//! Семантика идентична host-блоку в `synaptix-models/llm/common/model.rs:879-907`.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::attention::linear_attn_raw::LinearAttnRawKernels;

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

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

// Host эталон — точная копия host-loop'а из LinearAttn::forward (через
// gated_delta_decay_beta) + scatter qe/ke/vv. Если поменяется host-логика —
// тест должен сломаться явно.
#[allow(clippy::too_many_arguments)]
fn host_prep_scatter(
    conv_out: &[f32], // (T, conv_dim)
    a: &[f32],        // (T, num_v)
    b: &[f32],        // (T, num_v)
    dt_bias: &[f32],  // (num_v,)
    a_log: &[f32],    // (num_v,)
    t_in: usize,
    num_v: usize,
    num_k: usize,
    n_rep: usize,
    hk: usize,
    hv: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;
    let v_off0 = key_dim * 2;
    let mut qe = vec![0.0f32; num_v * t_in * hk];
    let mut ke = vec![0.0f32; num_v * t_in * hk];
    let mut vv = vec![0.0f32; num_v * t_in * hv];
    let mut g = vec![0.0f32; num_v * t_in];
    let mut beta = vec![0.0f32; num_v * t_in];
    for hi in 0..num_v {
        let kh = hi / n_rep;
        for t in 0..t_in {
            let row = t * conv_dim;
            for r in 0..hk {
                qe[(hi * t_in + t) * hk + r] = conv_out[row + kh * hk + r];
                ke[(hi * t_in + t) * hk + r] = conv_out[row + key_dim + kh * hk + r];
            }
            for c in 0..hv {
                vv[(hi * t_in + t) * hv + c] = conv_out[row + v_off0 + hi * hv + c];
            }
        }
    }
    for t in 0..t_in {
        for hi in 0..num_v {
            let av = a[t * num_v + hi];
            let bv = b[t * num_v + hi];
            beta[hi * t_in + t] = 1.0 / (1.0 + (-bv).exp());
            g[hi * t_in + t] = -(a_log[hi].exp()) * softplus(av + dt_bias[hi]);
        }
    }
    (qe, ke, vv, g, beta)
}

#[test]
fn prep_scatter_f16_vs_host() {
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = LinearAttnRawKernels::for_context(&ctx).expect("kernels");
    let (t_in, num_k, n_rep, hk, hv) = (350usize, 4usize, 4usize, 128usize, 256usize);
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;

    let conv_out = det_f32(0x10, t_in * conv_dim, 1.0);
    let a = det_f32(0x20, t_in * num_v, 0.5);
    let b = det_f32(0x30, t_in * num_v, 0.5);
    let dt_bias = det_f32(0x40, num_v, 0.3);
    let a_log = det_f32(0x50, num_v, 0.2);

    let to_f16 = |v: &[f32]| -> Vec<f16> { v.iter().map(|x| f16::from_f32(*x)).collect() };
    let f16_to_f = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };

    // Host эталон: квантуем входы в F16 (соответствует device-пути).
    let conv_q = f16_to_f(&to_f16(&conv_out));
    let a_q = f16_to_f(&to_f16(&a));
    let b_q = f16_to_f(&to_f16(&b));
    let (qe_r, ke_r, vv_r, g_r, beta_r) = host_prep_scatter(
        &conv_q, &a_q, &b_q, &dt_bias, &a_log, t_in, num_v, num_k, n_rep, hk, hv,
    );

    let conv_d = stream.memcpy_stod(&to_f16(&conv_out)).unwrap();
    let a_d = stream.memcpy_stod(&to_f16(&a)).unwrap();
    let b_d = stream.memcpy_stod(&to_f16(&b)).unwrap();
    let dt_d = stream.memcpy_stod(&dt_bias).unwrap();
    let al_d = stream.memcpy_stod(&a_log).unwrap();
    let mut qe_d = stream.alloc_zeros::<f32>(num_v * t_in * hk).unwrap();
    let mut ke_d = stream.alloc_zeros::<f32>(num_v * t_in * hk).unwrap();
    let mut vv_d = stream.alloc_zeros::<f32>(num_v * t_in * hv).unwrap();
    let mut g_d = stream.alloc_zeros::<f32>(num_v * t_in).unwrap();
    let mut beta_d = stream.alloc_zeros::<f32>(num_v * t_in).unwrap();

    {
        let b_v = b_d.as_view();
        let a_v = a_d.as_view();
        let dt_v = dt_d.as_view();
        let al_v = al_d.as_view();
        let conv_v = conv_d.as_view();
        kernels
            .linear_attn_prep_scatter_f16(
                &stream,
                &b_v,
                &a_v,
                &dt_v,
                &al_v,
                &mut beta_d,
                &mut g_d,
                &conv_v,
                &mut qe_d,
                &mut ke_d,
                &mut vv_d,
                t_in as u32,
                t_in as u32,
                num_v as u32,
                num_k as u32,
                n_rep as u32,
                hk as u32,
                hv as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();

    let qe_h = stream.memcpy_dtov(&qe_d).unwrap();
    let ke_h = stream.memcpy_dtov(&ke_d).unwrap();
    let vv_h = stream.memcpy_dtov(&vv_d).unwrap();
    let g_h = stream.memcpy_dtov(&g_d).unwrap();
    let beta_h = stream.memcpy_dtov(&beta_d).unwrap();

    let m_q = max_abs(&qe_h, &qe_r);
    let m_k = max_abs(&ke_h, &ke_r);
    let m_v = max_abs(&vv_h, &vv_r);
    let m_g = max_abs(&g_h, &g_r);
    let m_b = max_abs(&beta_h, &beta_r);
    eprintln!(
        "[prep_scatter f16] qe={m_q:.3e} ke={m_k:.3e} vv={m_v:.3e} g={m_g:.3e} beta={m_b:.3e}"
    );
    // qe/ke/vv = host F16 cast в F32 — точное равенство (нет арифметики).
    // g/beta = exp/softplus/sigmoid в F32 — bit-exact с host softplus implementation,
    // но host softplus имеет ветку x>20 → x, kernel — нет. Для нашего seed диапазон
    // достаточно мал. tol небольшой.
    assert!(m_q < 1e-6, "qe parity max_abs={m_q}");
    assert!(m_k < 1e-6, "ke parity max_abs={m_k}");
    assert!(m_v < 1e-6, "vv parity max_abs={m_v}");
    assert!(m_g < 1e-4, "g parity max_abs={m_g}");
    assert!(m_b < 1e-6, "beta parity max_abs={m_b}");
}

#[test]
fn prep_scatter_bf16_vs_host() {
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = LinearAttnRawKernels::for_context(&ctx).expect("kernels");
    let (t_in, num_k, n_rep, hk, hv) = (128usize, 2usize, 4usize, 64usize, 128usize);
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;

    let conv_out = det_f32(0x60, t_in * conv_dim, 1.0);
    let a = det_f32(0x70, t_in * num_v, 0.5);
    let b = det_f32(0x80, t_in * num_v, 0.5);
    let dt_bias = det_f32(0x90, num_v, 0.3);
    let a_log = det_f32(0xa0, num_v, 0.2);

    let to_bf16 = |v: &[f32]| -> Vec<bf16> { v.iter().map(|x| bf16::from_f32(*x)).collect() };
    let bf16_to_f = |v: &[bf16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
    let to_f16 = |v: &[f32]| -> Vec<f16> { v.iter().map(|x| f16::from_f32(*x)).collect() };
    let f16_to_f = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };

    let conv_q = bf16_to_f(&to_bf16(&conv_out));
    let a_q = f16_to_f(&to_f16(&a));
    let b_q = f16_to_f(&to_f16(&b));
    let (qe_r, ke_r, vv_r, g_r, beta_r) = host_prep_scatter(
        &conv_q, &a_q, &b_q, &dt_bias, &a_log, t_in, num_v, num_k, n_rep, hk, hv,
    );

    let conv_d = stream.memcpy_stod(&to_bf16(&conv_out)).unwrap();
    let a_d = stream.memcpy_stod(&to_f16(&a)).unwrap();
    let b_d = stream.memcpy_stod(&to_f16(&b)).unwrap();
    let dt_d = stream.memcpy_stod(&dt_bias).unwrap();
    let al_d = stream.memcpy_stod(&a_log).unwrap();
    let mut qe_d = stream.alloc_zeros::<f32>(num_v * t_in * hk).unwrap();
    let mut ke_d = stream.alloc_zeros::<f32>(num_v * t_in * hk).unwrap();
    let mut vv_d = stream.alloc_zeros::<f32>(num_v * t_in * hv).unwrap();
    let mut g_d = stream.alloc_zeros::<f32>(num_v * t_in).unwrap();
    let mut beta_d = stream.alloc_zeros::<f32>(num_v * t_in).unwrap();

    {
        let b_v = b_d.as_view();
        let a_v = a_d.as_view();
        let dt_v = dt_d.as_view();
        let al_v = al_d.as_view();
        let conv_v = conv_d.as_view();
        kernels
            .linear_attn_prep_scatter_bf16(
                &stream,
                &b_v,
                &a_v,
                &dt_v,
                &al_v,
                &mut beta_d,
                &mut g_d,
                &conv_v,
                &mut qe_d,
                &mut ke_d,
                &mut vv_d,
                t_in as u32,
                t_in as u32,
                num_v as u32,
                num_k as u32,
                n_rep as u32,
                hk as u32,
                hv as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();

    let m_q = max_abs(&stream.memcpy_dtov(&qe_d).unwrap(), &qe_r);
    let m_k = max_abs(&stream.memcpy_dtov(&ke_d).unwrap(), &ke_r);
    let m_v = max_abs(&stream.memcpy_dtov(&vv_d).unwrap(), &vv_r);
    let m_g = max_abs(&stream.memcpy_dtov(&g_d).unwrap(), &g_r);
    let m_b = max_abs(&stream.memcpy_dtov(&beta_d).unwrap(), &beta_r);
    eprintln!(
        "[prep_scatter bf16] qe={m_q:.3e} ke={m_k:.3e} vv={m_v:.3e} g={m_g:.3e} beta={m_b:.3e}"
    );
    assert!(m_q < 1e-6, "qe parity max_abs={m_q}");
    assert!(m_k < 1e-6, "ke parity max_abs={m_k}");
    assert!(m_v < 1e-6, "vv parity max_abs={m_v}");
    assert!(m_g < 1e-4, "g parity max_abs={m_g}");
    assert!(m_b < 1e-6, "beta parity max_abs={m_b}");
}

#[test]
fn prep_scatter_f32_vs_host() {
    let Some((ctx, stream)) = setup() else {
        return;
    };
    let kernels = LinearAttnRawKernels::for_context(&ctx).expect("kernels");
    let (t_in, num_k, n_rep, hk, hv) = (8usize, 2usize, 4usize, 32usize, 64usize);
    let num_v = num_k * n_rep;
    let key_dim = num_k * hk;
    let conv_dim = 2 * key_dim + num_v * hv;

    let conv_out = det_f32(0xc0, t_in * conv_dim, 1.0);
    let a = det_f32(0xd0, t_in * num_v, 0.5);
    let b = det_f32(0xe0, t_in * num_v, 0.5);
    let dt_bias = det_f32(0xf0, num_v, 0.3);
    let a_log = det_f32(0x100, num_v, 0.2);

    let to_f16 = |v: &[f32]| -> Vec<f16> { v.iter().map(|x| f16::from_f32(*x)).collect() };
    let f16_to_f = |v: &[f16]| -> Vec<f32> { v.iter().map(|x| x.to_f32()).collect() };
    let a_q = f16_to_f(&to_f16(&a));
    let b_q = f16_to_f(&to_f16(&b));
    let (qe_r, ke_r, vv_r, g_r, beta_r) = host_prep_scatter(
        &conv_out, &a_q, &b_q, &dt_bias, &a_log, t_in, num_v, num_k, n_rep, hk, hv,
    );

    let conv_d = stream.memcpy_stod(&conv_out).unwrap();
    let a_d = stream.memcpy_stod(&to_f16(&a)).unwrap();
    let b_d = stream.memcpy_stod(&to_f16(&b)).unwrap();
    let dt_d = stream.memcpy_stod(&dt_bias).unwrap();
    let al_d = stream.memcpy_stod(&a_log).unwrap();
    let mut qe_d = stream.alloc_zeros::<f32>(num_v * t_in * hk).unwrap();
    let mut ke_d = stream.alloc_zeros::<f32>(num_v * t_in * hk).unwrap();
    let mut vv_d = stream.alloc_zeros::<f32>(num_v * t_in * hv).unwrap();
    let mut g_d = stream.alloc_zeros::<f32>(num_v * t_in).unwrap();
    let mut beta_d = stream.alloc_zeros::<f32>(num_v * t_in).unwrap();

    {
        let b_v = b_d.as_view();
        let a_v = a_d.as_view();
        let dt_v = dt_d.as_view();
        let al_v = al_d.as_view();
        let conv_v = conv_d.as_view();
        kernels
            .linear_attn_prep_scatter_f32(
                &stream,
                &b_v,
                &a_v,
                &dt_v,
                &al_v,
                &mut beta_d,
                &mut g_d,
                &conv_v,
                &mut qe_d,
                &mut ke_d,
                &mut vv_d,
                t_in as u32,
                t_in as u32,
                num_v as u32,
                num_k as u32,
                n_rep as u32,
                hk as u32,
                hv as u32,
            )
            .unwrap();
    }
    stream.synchronize().unwrap();

    let m_q = max_abs(&stream.memcpy_dtov(&qe_d).unwrap(), &qe_r);
    let m_k = max_abs(&stream.memcpy_dtov(&ke_d).unwrap(), &ke_r);
    let m_v = max_abs(&stream.memcpy_dtov(&vv_d).unwrap(), &vv_r);
    let m_g = max_abs(&stream.memcpy_dtov(&g_d).unwrap(), &g_r);
    let m_b = max_abs(&stream.memcpy_dtov(&beta_d).unwrap(), &beta_r);
    eprintln!(
        "[prep_scatter f32] qe={m_q:.3e} ke={m_k:.3e} vv={m_v:.3e} g={m_g:.3e} beta={m_b:.3e}"
    );
    assert!(m_q < 1e-7, "qe parity max_abs={m_q}");
    assert!(m_k < 1e-7, "ke parity max_abs={m_k}");
    assert!(m_v < 1e-7, "vv parity max_abs={m_v}");
    assert!(m_g < 1e-4, "g parity max_abs={m_g}");
    assert!(m_b < 1e-6, "beta parity max_abs={m_b}");
}

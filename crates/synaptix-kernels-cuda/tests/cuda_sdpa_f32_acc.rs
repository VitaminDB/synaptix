#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::attention::sdpa_f32_acc::{
    sdpa_f32_acc_bf16, sdpa_f32_acc_f16, sdpa_f32_acc_f32, SdpaF32AccKernels,
};

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
            ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn cpu_sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    nh: usize,
    nkv: usize,
    t_q: usize,
    t_kv: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let n_rep = nh / nkv;
    let mut out = vec![0.0_f32; b * nh * t_q * d];
    for bi in 0..b {
        for h in 0..nh {
            let h_kv = h / n_rep;
            for ti in 0..t_q {
                let q_pos = if t_kv >= t_q { t_kv - t_q + ti } else { ti };
                let mut scores = vec![0.0_f32; t_kv];
                for j in 0..t_kv {
                    if causal && j > q_pos {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let q_off = ((bi * nh + h) * t_q + ti) * d;
                    let k_off = ((bi * nkv + h_kv) * t_kv + j) * d;
                    let mut s = 0.0_f32;
                    for kk in 0..d {
                        s += q[q_off + kk] * k[k_off + kk];
                    }
                    scores[j] = s * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0_f32;
                let mut es = vec![0.0_f32; t_kv];
                for j in 0..t_kv {
                    if scores[j].is_finite() {
                        let e = (scores[j] - m).exp();
                        es[j] = e;
                        l += e;
                    }
                }
                for kk in 0..d {
                    let mut acc = 0.0_f32;
                    for j in 0..t_kv {
                        if es[j] > 0.0 {
                            let v_off = ((bi * nkv + h_kv) * t_kv + j) * d;
                            acc += es[j] * v[v_off + kk];
                        }
                    }
                    out[((bi * nh + h) * t_q + ti) * d + kk] = if l > 0.0 { acc / l } else { 0.0 };
                }
            }
        }
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn sdpa_f32_gqa_noncausal() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SdpaF32AccKernels::for_context(&ctx).expect("compile sdpa_f32_acc");
    let (b, nh, nkv, tq, tkv, d) = (2usize, 8usize, 2usize, 16usize, 48usize, 64usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = det_f32(0x5A1, b * nh * tq * d, 0.3);
    let k = det_f32(0x5B2, b * nkv * tkv * d, 0.3);
    let v = det_f32(0x5C3, b * nkv * tkv * d, 0.3);
    let exp = cpu_sdpa(&q, &k, &v, b, nh, nkv, tq, tkv, d, scale, false);

    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    sdpa_f32_acc_f32(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, false,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let m = max_abs(&got, &exp);
    eprintln!("[sdpa_f32 gqa noncausal] max_abs={m:.7}");
    assert!(m < 1e-4, "max_abs={m}");
}

#[test]
fn sdpa_f32_causal() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SdpaF32AccKernels::for_context(&ctx).expect("compile sdpa_f32_acc");
    let (b, nh, nkv, tq, tkv, d) = (1usize, 4usize, 4usize, 32usize, 32usize, 64usize);
    let scale = 1.0 / (d as f32).sqrt();
    let q = det_f32(0x6A1, b * nh * tq * d, 0.3);
    let k = det_f32(0x6B2, b * nkv * tkv * d, 0.3);
    let v = det_f32(0x6C3, b * nkv * tkv * d, 0.3);
    let exp = cpu_sdpa(&q, &k, &v, b, nh, nkv, tq, tkv, d, scale, true);

    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    sdpa_f32_acc_f32(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, true,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let m = max_abs(&got, &exp);
    eprintln!("[sdpa_f32 causal] max_abs={m:.7}");
    assert!(m < 1e-4, "max_abs={m}");
}

#[test]
fn sdpa_f16_causal() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SdpaF32AccKernels::for_context(&ctx).expect("compile sdpa_f32_acc");
    let (b, nh, nkv, tq, tkv, d) = (2usize, 8usize, 2usize, 24usize, 24usize, 64usize);
    let scale = 1.0 / (d as f32).sqrt();
    let qf = det_f32(0x7A1, b * nh * tq * d, 0.3);
    let kf = det_f32(0x7B2, b * nkv * tkv * d, 0.3);
    let vf = det_f32(0x7C3, b * nkv * tkv * d, 0.3);
    let q: Vec<f16> = qf.iter().map(|x| f16::from_f32(*x)).collect();
    let k: Vec<f16> = kf.iter().map(|x| f16::from_f32(*x)).collect();
    let v: Vec<f16> = vf.iter().map(|x| f16::from_f32(*x)).collect();
    let exp = cpu_sdpa(
        &q.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &k.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &v.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        b,
        nh,
        nkv,
        tq,
        tkv,
        d,
        scale,
        true,
    );
    let dq: CudaSlice<f16> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f16> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f16> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<f16> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    sdpa_f32_acc_f16(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, true,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gh: Vec<f16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = gh.iter().map(|x| x.to_f32()).collect();
    let m = max_abs(&got, &exp);
    eprintln!("[sdpa_f16 causal] max_abs={m:.4}");
    assert!(m < 0.02, "max_abs={m}");
}

#[test]
fn sdpa_bf16_noncausal() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = SdpaF32AccKernels::for_context(&ctx).expect("compile sdpa_f32_acc");
    let (b, nh, nkv, tq, tkv, d) = (1usize, 4usize, 1usize, 20usize, 40usize, 64usize);
    let scale = 1.0 / (d as f32).sqrt();
    let qf = det_f32(0x8A1, b * nh * tq * d, 0.3);
    let kf = det_f32(0x8B2, b * nkv * tkv * d, 0.3);
    let vf = det_f32(0x8C3, b * nkv * tkv * d, 0.3);
    let q: Vec<bf16> = qf.iter().map(|x| bf16::from_f32(*x)).collect();
    let k: Vec<bf16> = kf.iter().map(|x| bf16::from_f32(*x)).collect();
    let v: Vec<bf16> = vf.iter().map(|x| bf16::from_f32(*x)).collect();
    let exp = cpu_sdpa(
        &q.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &k.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &v.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        b,
        nh,
        nkv,
        tq,
        tkv,
        d,
        scale,
        false,
    );
    let dq: CudaSlice<bf16> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<bf16> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<bf16> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<bf16> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    sdpa_f32_acc_bf16(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, false,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gb: Vec<bf16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = gb.iter().map(|x| x.to_f32()).collect();
    let m = max_abs(&got, &exp);
    eprintln!("[sdpa_bf16 noncausal] max_abs={m:.4}");
    assert!(m < 0.05, "max_abs={m}");
}

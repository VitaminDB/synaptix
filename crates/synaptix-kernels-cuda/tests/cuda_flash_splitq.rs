//! Гейты flash_splitq (FA-5, единственное tensor-core prefill-ядро после
//! удаления flash_v4): корректность против CPU-SDPA-эталона (F32 softmax)
//! на GQA/MHA, causal/non, hd 64/128/256, малые Tq (бывший v4-фоллбэк),
//! bshd-layout (SDXL) и dev-вариант (device-Tkv, CUDA-graph prefill).
#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_core::dtype::DType;
use synaptix_kernels_cuda::attention::flash_splitq::{
    flash_splitq_u8, flash_splitq_u8_dev, FlashSplitQKernels,
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

// CPU-эталон: двухпроходный F32-softmax SDPA (как cuda_sdpa_f32_acc).
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
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn htod_f16(stream: &Arc<CudaStream>, data: &[f32], dt: DType) -> CudaSlice<u8> {
    let bytes: Vec<u8> = match dt {
        DType::F16 => data.iter().flat_map(|x| f16::from_f32(*x).to_le_bytes()).collect(),
        DType::BF16 => data.iter().flat_map(|x| bf16::from_f32(*x).to_le_bytes()).collect(),
        _ => unreachable!(),
    };
    stream.clone_htod(&bytes).unwrap()
}

fn dtoh_f32(stream: &Arc<CudaStream>, buf: &CudaSlice<u8>, n: usize, dt: DType) -> Vec<f32> {
    let bytes: Vec<u8> = stream.clone_dtoh(buf).unwrap();
    (0..n)
        .map(|i| match dt {
            DType::F16 => f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f32(),
            DType::BF16 => bf16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]).to_f32(),
            _ => unreachable!(),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn case(
    b: usize,
    nh: usize,
    nkv: usize,
    tq: usize,
    tkv: usize,
    d: usize,
    dt: DType,
    causal: bool,
    tol: f32,
    tag: &str,
) {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashSplitQKernels::for_context(&ctx).expect("compile flash_splitq");
    let scale = 1.0 / (d as f32).sqrt();
    // эталон считаем на ОКРУГЛЁННЫХ к dt значениях (как ядро видит данные)
    let rnd = |v: Vec<f32>| -> Vec<f32> {
        match dt {
            DType::F16 => v.iter().map(|x| f16::from_f32(*x).to_f32()).collect(),
            DType::BF16 => v.iter().map(|x| bf16::from_f32(*x).to_f32()).collect(),
            _ => unreachable!(),
        }
    };
    let qf = rnd(det_f32(0x11A, b * nh * tq * d, 0.3));
    let kf = rnd(det_f32(0x22B, b * nkv * tkv * d, 0.3));
    let vf = rnd(det_f32(0x33C, b * nkv * tkv * d, 0.3));
    let exp = cpu_sdpa(&qf, &kf, &vf, b, nh, nkv, tq, tkv, d, scale, causal);

    let dq = htod_f16(&stream, &qf, dt);
    let dk = htod_f16(&stream, &kf, dt);
    let dv = htod_f16(&stream, &vf, dt);
    let out_n = b * nh * tq * d;
    let mut dout: CudaSlice<u8> = stream.alloc_zeros(out_n * 2).unwrap();
    flash_splitq_u8(
        &kernels, &stream, &dq, 0, &dk, 0, &dv, 0, &mut dout, 0, b as u32, nh as u32, nkv as u32,
        tq as u32, tkv as u32, d as u32, scale, causal, 0, dt, false,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got = dtoh_f32(&stream, &dout, out_n, dt);
    let m = max_abs(&got, &exp);
    eprintln!("[splitq {tag}] max_abs={m:.4}");
    assert!(m < tol, "{tag}: max_abs={m}");
}

#[test]
fn splitq_f16_causal_gqa_hd256() {
    case(2, 8, 2, 48, 80, 256, DType::F16, true, 0.03, "f16 causal gqa hd256");
}
#[test]
fn splitq_f16_noncausal_mha_hd128() {
    case(1, 4, 4, 16, 64, 128, DType::F16, false, 0.03, "f16 noncausal hd128");
}
#[test]
fn splitq_bf16_causal_hd128() {
    case(2, 8, 2, 100, 164, 128, DType::BF16, true, 0.08, "bf16 causal hd128");
}
#[test]
fn splitq_f16_hd64() {
    case(1, 8, 8, 96, 96, 64, DType::F16, true, 0.03, "f16 causal hd64");
}
#[test]
fn splitq_small_tq() {
    // бывший v4-фоллбэк (Tq 2..63) теперь идёт в splitq: ceil-грид BM=64.
    case(1, 8, 2, 2, 512, 128, DType::F16, true, 0.03, "f16 tq=2");
    case(1, 8, 2, 33, 512, 128, DType::BF16, true, 0.08, "bf16 tq=33");
}

#[test]
fn splitq_dev_matches_host_tkv() {
    // dev-вариант (Tkv из device-буфера) == host-вариант бит-в-бит.
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashSplitQKernels::for_context(&ctx).expect("compile flash_splitq");
    let (b, nh, nkv, tq, tkv, cap, d) = (1usize, 8usize, 2usize, 64usize, 192usize, 256usize, 128usize);
    let dt = DType::F16;
    let scale = 1.0 / (d as f32).sqrt();
    let qf = det_f32(0x77A, b * nh * tq * d, 0.3);
    // KV в préalloc-буфере ёмкостью cap (t_stride=cap), активны первые tkv
    let kf = det_f32(0x88B, b * nkv * cap * d, 0.3);
    let vf = det_f32(0x99C, b * nkv * cap * d, 0.3);
    let dq = htod_f16(&stream, &qf, dt);
    let dk = htod_f16(&stream, &kf, dt);
    let dv = htod_f16(&stream, &vf, dt);
    let out_n = b * nh * tq * d;
    let mut out_host: CudaSlice<u8> = stream.alloc_zeros(out_n * 2).unwrap();
    let mut out_dev: CudaSlice<u8> = stream.alloc_zeros(out_n * 2).unwrap();
    flash_splitq_u8(
        &kernels, &stream, &dq, 0, &dk, 0, &dv, 0, &mut out_host, 0, b as u32, nh as u32,
        nkv as u32, tq as u32, tkv as u32, d as u32, scale, true, cap as u32, dt, false,
    )
    .unwrap();
    let tc: CudaSlice<u8> = stream.clone_htod(&(tkv as i32).to_le_bytes()).unwrap();
    flash_splitq_u8_dev(
        &kernels, &stream, &dq, 0, &dk, 0, &dv, 0, &mut out_dev, 0, &tc, 0, b as u32, nh as u32,
        nkv as u32, tq as u32, d as u32, scale, true, cap as u32, dt,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let a: Vec<u8> = stream.clone_dtoh(&out_host).unwrap();
    let bts: Vec<u8> = stream.clone_dtoh(&out_dev).unwrap();
    assert_eq!(a, bts, "dev-вариант разошёлся с host-Tkv");
}

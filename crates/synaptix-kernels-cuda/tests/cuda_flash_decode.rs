#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use synaptix_kernels_cuda::attention::flash_decode::{
    flash_decode_bf16, flash_decode_f16, flash_decode_f32, FlashDecodeKernels,
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

// CPU-эталон: точный двухпроходный softmax-attention в F32 (== cpu_sdpa из
// cuda_sdpa_f32_acc.rs). GQA + causal, layout q[B,NH,Tq,D] / k,v[B,NKV,Tkv,D].
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

// ── F32: decode T_q=1, GQA, split_k=8, длинный Tkv (несколько KV-тайлов на сегмент) ──
#[test]
fn flash_decode_f32_gqa_decode() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (2usize, 8usize, 2usize, 1usize, 600usize, 64usize);
    let split_k = 8u32;
    let scale = 1.0 / (d as f32).sqrt();
    let q = det_f32(0x5A1, b * nh * tq * d, 0.3);
    let k = det_f32(0x5B2, b * nkv * tkv * d, 0.3);
    let v = det_f32(0x5C3, b * nkv * tkv * d, 0.3);
    let exp = cpu_sdpa(&q, &k, &v, b, nh, nkv, tq, tkv, d, scale, true);

    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    flash_decode_f32(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, true, split_k,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode f32 gqa decode] max_abs={m:.7}");
    assert!(m < 1e-4, "max_abs={m}");
}

// ── F32: split_k > число тайлов (пустые сегменты), D=128, causal ──
#[test]
fn flash_decode_f32_sparse_splits() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (1usize, 4usize, 4usize, 1usize, 40usize, 128usize);
    let split_k = 16u32; // seg = ceil(40/16) = 3 → последние сегменты пустые
    let scale = 1.0 / (d as f32).sqrt();
    let q = det_f32(0x6A1, b * nh * tq * d, 0.3);
    let k = det_f32(0x6B2, b * nkv * tkv * d, 0.3);
    let v = det_f32(0x6C3, b * nkv * tkv * d, 0.3);
    let exp = cpu_sdpa(&q, &k, &v, b, nh, nkv, tq, tkv, d, scale, true);

    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    flash_decode_f32(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, true, split_k,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode f32 sparse splits] max_abs={m:.7}");
    assert!(m < 1e-4, "max_abs={m}");
}

// ── F32: T_q>1 (chunked) + causal, split_k=4 — проверка causal-маски в split ──
#[test]
fn flash_decode_f32_chunk_causal() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (1usize, 4usize, 2usize, 8usize, 200usize, 64usize);
    let split_k = 4u32;
    let scale = 1.0 / (d as f32).sqrt();
    let q = det_f32(0x9A1, b * nh * tq * d, 0.3);
    let k = det_f32(0x9B2, b * nkv * tkv * d, 0.3);
    let v = det_f32(0x9C3, b * nkv * tkv * d, 0.3);
    let exp = cpu_sdpa(&q, &k, &v, b, nh, nkv, tq, tkv, d, scale, true);

    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    flash_decode_f32(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, true, split_k,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode f32 chunk causal] max_abs={m:.7}");
    assert!(m < 1e-4, "max_abs={m}");
}

// ── F16 decode ──
#[test]
fn flash_decode_f16_decode() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (2usize, 8usize, 2usize, 1usize, 320usize, 128usize);
    let split_k = 8u32;
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
    flash_decode_f16(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, true, split_k,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gh: Vec<f16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = gh.iter().map(|x| x.to_f32()).collect();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode f16 decode] max_abs={m:.4}");
    assert!(m < 0.02, "max_abs={m}");
}

// ── BF16 decode ──
#[test]
fn flash_decode_bf16_decode() {
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (1usize, 4usize, 1usize, 1usize, 257usize, 64usize);
    let split_k = 6u32;
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
    flash_decode_bf16(
        &kernels, &stream, &dq, &dk, &dv, &mut dout, b as u32, nh as u32, nkv as u32, tq as u32,
        tkv as u32, d as u32, scale, false, split_k,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let gb: Vec<bf16> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = gb.iter().map(|x| x.to_f32()).collect();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode bf16 decode] max_abs={m:.4}");
    assert!(m < 0.05, "max_abs={m}");
}

// ── Device-resident length (CUDA-graph path): flash_decode_u8_dev читает t_kv
//    из device-памяти, strided буфер. Сверка с cpu_sdpa на той же активной длине;
//    проверяет, что dev-вариант численно идентичен immediate-пути. ──
#[test]
fn flash_decode_f16_dev_length() {
    use synaptix_core::dtype::DType;
    use synaptix_kernels_cuda::attention::flash_decode::flash_decode_u8_dev;
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (1usize, 8usize, 2usize, 1usize, 300usize, 128usize);
    let max_seq = 512usize;
    let split_k = 8u32;
    let scale = 1.0 / (d as f32).sqrt();
    let qf = det_f32(0xA71A, b * nh * tq * d, 0.3);
    let kf = det_f32(0xA72B, b * nkv * tkv * d, 0.3);
    let vf = det_f32(0xA73C, b * nkv * tkv * d, 0.3);
    let q: Vec<f16> = qf.iter().map(|x| f16::from_f32(*x)).collect();
    let k_act: Vec<f16> = kf.iter().map(|x| f16::from_f32(*x)).collect();
    let v_act: Vec<f16> = vf.iter().map(|x| f16::from_f32(*x)).collect();
    let exp = cpu_sdpa(
        &q.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &k_act.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        &v_act.iter().map(|x| x.to_f32()).collect::<Vec<_>>(),
        b,
        nh,
        nkv,
        tq,
        tkv,
        d,
        scale,
        true,
    );

    // Physical буфер [b,nkv,max_seq,d] с мусором; активные строки [0:tkv].
    let mut k_phys: Vec<f16> = det_f32(0xDEAD, b * nkv * max_seq * d, 9.0)
        .iter()
        .map(|x| f16::from_f32(*x))
        .collect();
    let mut v_phys: Vec<f16> = det_f32(0xBEEF, b * nkv * max_seq * d, 9.0)
        .iter()
        .map(|x| f16::from_f32(*x))
        .collect();
    for bh in 0..(b * nkv) {
        for t in 0..tkv {
            for dd in 0..d {
                k_phys[(bh * max_seq + t) * d + dd] = k_act[(bh * tkv + t) * d + dd];
                v_phys[(bh * max_seq + t) * d + dd] = v_act[(bh * tkv + t) * d + dd];
            }
        }
    }
    let f16_to_u8 = |v: &[f16]| -> Vec<u8> {
        let mut o = Vec::with_capacity(v.len() * 2);
        for x in v {
            o.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        o
    };
    let dq: CudaSlice<u8> = stream.clone_htod(&f16_to_u8(&q)).unwrap();
    let dk: CudaSlice<u8> = stream.clone_htod(&f16_to_u8(&k_phys)).unwrap();
    let dv: CudaSlice<u8> = stream.clone_htod(&f16_to_u8(&v_phys)).unwrap();
    let mut dout: CudaSlice<u8> = stream.alloc_zeros(b * nh * tq * d * 2).unwrap();
    let tkv_dev: CudaSlice<u32> = stream.clone_htod(&[tkv as u32]).unwrap();

    flash_decode_u8_dev(
        &kernels,
        &stream,
        &dq,
        0,
        &dk,
        0,
        &dv,
        0,
        &mut dout,
        0,
        b as u32,
        nh as u32,
        nkv as u32,
        tq as u32,
        &tkv_dev.as_view(),
        d as u32,
        scale,
        true,
        split_k,
        max_seq as u32,
        DType::F16,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let raw: Vec<u8> = stream.clone_dtoh(&dout).unwrap();
    let got: Vec<f32> = raw
        .chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode f16 dev length] max_abs={m:.4}");
    assert!(m < 0.02, "max_abs={m}");
}

// ── F32: strided KV-буфер (t_stride=max_seq > t_kv). Активные строки [0:t_kv]
//    в preallocated буфере [b,nkv,max_seq,d], остальное — мусор. Kernel должен
//    читать только активные строки через t_stride и игнорировать padding. ──
#[test]
fn flash_decode_f32_strided_buffer() {
    use synaptix_core::dtype::DType;
    use synaptix_kernels_cuda::attention::flash_decode::flash_decode;
    let Some((ctx, stream)) = setup() else { return };
    let kernels = FlashDecodeKernels::for_context(&ctx).expect("compile flash_decode");
    let (b, nh, nkv, tq, tkv, d) = (2usize, 8usize, 2usize, 1usize, 300usize, 128usize);
    let max_seq = 512usize;
    let split_k = 8u32;
    let scale = 1.0 / (d as f32).sqrt();
    let q = det_f32(0x71A, b * nh * tq * d, 0.3);
    let k_act = det_f32(0x72B, b * nkv * tkv * d, 0.3);
    let v_act = det_f32(0x73C, b * nkv * tkv * d, 0.3);
    let exp = cpu_sdpa(&q, &k_act, &v_act, b, nh, nkv, tq, tkv, d, scale, true);

    // Physical буфер [b,nkv,max_seq,d] с мусором; активные строки [0:tkv] = k_act.
    let mut k_phys = det_f32(0xDEAD, b * nkv * max_seq * d, 9.0);
    let mut v_phys = det_f32(0xBEEF, b * nkv * max_seq * d, 9.0);
    for bh in 0..(b * nkv) {
        for t in 0..tkv {
            for dd in 0..d {
                k_phys[(bh * max_seq + t) * d + dd] = k_act[(bh * tkv + t) * d + dd];
                v_phys[(bh * max_seq + t) * d + dd] = v_act[(bh * tkv + t) * d + dd];
            }
        }
    }
    let dq: CudaSlice<f32> = stream.clone_htod(&q).unwrap();
    let dk: CudaSlice<f32> = stream.clone_htod(&k_phys).unwrap();
    let dv: CudaSlice<f32> = stream.clone_htod(&v_phys).unwrap();
    let mut dout: CudaSlice<f32> = stream.alloc_zeros(b * nh * tq * d).unwrap();
    flash_decode::<f32>(
        &kernels,
        &stream,
        &dq,
        &dk,
        &dv,
        &mut dout,
        b as u32,
        nh as u32,
        nkv as u32,
        tq as u32,
        tkv as u32,
        d as u32,
        scale,
        true,
        split_k,
        max_seq as u32,
        DType::F32,
    )
    .unwrap();
    stream.synchronize().unwrap();
    let got: Vec<f32> = stream.clone_dtoh(&dout).unwrap();
    let m = max_abs(&got, &exp);
    eprintln!("[flash_decode f32 strided buffer] max_abs={m:.7}");
    assert!(m < 1e-4, "max_abs={m}");
}

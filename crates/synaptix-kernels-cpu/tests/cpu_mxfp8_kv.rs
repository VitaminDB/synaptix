//! MXFP8-KV (Blackwell block-scale) CPU path: per-32-block квантизующий append +
//! block-scale dequant flash-attention.
//!
//! Замкнутый цикл (как cpu_fp8_kv, но per-32-block E8M0): эталонные K/V квантизуются
//! тем же блочным E8M0/E4M3 кодеком, что и backend → f32-SDPA на деквантизованных
//! значениях сверяется с `Tensor::flash_attention_mxfp8kv`. Совпадение доказывает,
//! что quant-append записал MXFP8+E8M0-scale корректно И что attention деквантизует
//! per-32-block верно (главная ловушка: scale зависит от d/32, не выносится из dot).

use half::bf16;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::quant::{decode_e4m3, e8m0_decode, e8m0_scale_byte, encode_e4m3, MXFP8_BLOCK};

fn det(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            ((u as f32 / u32::MAX as f32) * 2.0 - 1.0) * scale
        })
        .collect()
}

// Per-(b,kv,token)-per-32-block amax→E8M0→encode→decode на f32-эталоне `[B,nkv,T,hd]`.
fn quantize_roundtrip_block(x: &[f32], b: usize, nkv: usize, t: usize, hd: usize) -> Vec<f32> {
    assert_eq!(hd % MXFP8_BLOCK, 0);
    let nb = hd / MXFP8_BLOCK;
    let mut out = vec![0.0f32; x.len()];
    for bh in 0..(b * nkv) {
        for ti in 0..t {
            let row = (bh * t + ti) * hd;
            for blk in 0..nb {
                let base = row + blk * MXFP8_BLOCK;
                let mut amax = 0.0f32;
                for i in 0..MXFP8_BLOCK {
                    amax = amax.max(x[base + i].abs());
                }
                let sv = e8m0_decode(e8m0_scale_byte(amax));
                for i in 0..MXFP8_BLOCK {
                    out[base + i] = decode_e4m3(encode_e4m3(x[base + i] / sv)) * sv;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn cpu_sdpa(
    q: &[f32], k: &[f32], v: &[f32],
    b: usize, nh: usize, nkv: usize, t_q: usize, t_kv: usize, d: usize,
    scale: f32, causal: bool,
) -> Vec<f32> {
    let n_rep = nh / nkv;
    let mut out = vec![0.0f32; b * nh * t_q * d];
    for bi in 0..b {
        for h in 0..nh {
            let h_kv = h / n_rep;
            for ti in 0..t_q {
                let q_pos = if t_kv >= t_q { t_kv - t_q + ti } else { ti };
                let mut scores = vec![f32::NEG_INFINITY; t_kv];
                for j in 0..t_kv {
                    if causal && j > q_pos {
                        continue;
                    }
                    let qo = ((bi * nh + h) * t_q + ti) * d;
                    let ko = ((bi * nkv + h_kv) * t_kv + j) * d;
                    let mut s = 0.0f32;
                    for kk in 0..d {
                        s += q[qo + kk] * k[ko + kk];
                    }
                    scores[j] = s * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0f32;
                let mut es = vec![0.0f32; t_kv];
                for j in 0..t_kv {
                    if scores[j].is_finite() {
                        let e = (scores[j] - m).exp();
                        es[j] = e;
                        l += e;
                    }
                }
                for kk in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..t_kv {
                        if es[j] > 0.0 {
                            let vo = ((bi * nkv + h_kv) * t_kv + j) * d;
                            acc += es[j] * v[vo + kk];
                        }
                    }
                    out[((bi * nh + h) * t_q + ti) * d + kk] = if l > 0.0 { acc / l } else { 0.0 };
                }
            }
        }
    }
    out
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na * nb) }
}

fn to_bf16(v: &[f32]) -> Vec<bf16> {
    v.iter().map(|&x| bf16::from_f32(x)).collect()
}

#[test]
fn mxfp8_kv_prefill_matches_block_quantized_reference() {
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;
    // hd=64 (= 2 блока по 32), GQA nh=4/nkv=2.
    let (b, nh, nkv, t, hd) = (1usize, 4, 2, 6, 64);
    let nb = hd / MXFP8_BLOCK;
    let max_seq = 16usize;
    let scale = 1.0 / (hd as f32).sqrt();

    let q_f = det(1, b * nh * t * hd, 1.0);
    let k_f = det(2, b * nkv * t * hd, 1.0);
    let v_f = det(3, b * nkv * t * hd, 1.0);

    let q = Tensor::from_vec(to_bf16(&q_f), vec![b, nh, t, hd], dev).unwrap();
    let k = Tensor::from_vec(to_bf16(&k_f), vec![b, nkv, t, hd], dev).unwrap();
    let v = Tensor::from_vec(to_bf16(&v_f), vec![b, nkv, t, hd], dev).unwrap();

    let mut k_buf = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut v_buf = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut k_sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();
    let mut v_sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();

    k_buf.kv_append_quant_mxfp8_inplace(&mut k_sc, &k, 0).unwrap();
    v_buf.kv_append_quant_mxfp8_inplace(&mut v_sc, &v, 0).unwrap();

    let k_q = k_buf.narrow(2, 0, t).unwrap();
    let v_q = v_buf.narrow(2, 0, t).unwrap();
    let ks = k_sc.narrow(2, 0, t).unwrap();
    let vs = v_sc.narrow(2, 0, t).unwrap();

    let out = q.flash_attention_mxfp8kv(&k_q, &v_q, &ks, &vs, scale, true).unwrap();
    let out_f: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .reshape(vec![b * nh * t * hd])
        .unwrap()
        .to_vec1()
        .unwrap();

    let q_b: Vec<f32> = to_bf16(&q_f).iter().map(|x| x.to_f32()).collect();
    let k_b: Vec<f32> = to_bf16(&k_f).iter().map(|x| x.to_f32()).collect();
    let v_b: Vec<f32> = to_bf16(&v_f).iter().map(|x| x.to_f32()).collect();
    let k_deq = quantize_roundtrip_block(&k_b, b, nkv, t, hd);
    let v_deq = quantize_roundtrip_block(&v_b, b, nkv, t, hd);
    let reference = cpu_sdpa(&q_b, &k_deq, &v_deq, b, nh, nkv, t, t, hd, scale, true);

    let cs = cos_sim(&out_f, &reference);
    let max_abs = out_f.iter().zip(&reference).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
    assert!(cs > 0.999, "cos_sim {cs} too low (max_abs {max_abs})");
}

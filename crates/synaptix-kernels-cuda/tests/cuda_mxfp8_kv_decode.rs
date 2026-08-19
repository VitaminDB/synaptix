
//! MXFP8-KV flash (CUDA): append-квант + per-32-block inline-dequant flash-attention.
//! Сверяет полный путь (Tensor::kv_append_quant_mxfp8_inplace +
//! flash_attention_mxfp8kv) с CPU-SDPA на block-деквантизованных K/V. Покрывает
//! decode (Tq=1) и prefill (Tq=T, scalar-путь). cos>0.999.

use half::bf16;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_kernels_cpu::quant::{decode_e4m3, e8m0_decode, e8m0_scale_byte, encode_e4m3, MXFP8_BLOCK};

fn setup() -> bool {
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

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

fn quantize_roundtrip_block(x: &[f32], b: usize, nkv: usize, t: usize, hd: usize) -> Vec<f32> {
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

// Прогон: append t_kv токенов в MXFP8-кеш, flash для q (t_q токенов), сверка с CPU.
fn run(b: usize, nh: usize, nkv: usize, t_q: usize, t_kv: usize, hd: usize, label: &str) {
    run_ms(b, nh, nkv, t_q, t_kv, hd, 64, label)
}

#[allow(clippy::too_many_arguments)]
fn run_ms(b: usize, nh: usize, nkv: usize, t_q: usize, t_kv: usize, hd: usize, max_seq: usize, label: &str) {
    let dev = Device::Cuda(0);
    let nb = hd / MXFP8_BLOCK;
    let scale = 1.0 / (hd as f32).sqrt();

    let q_f = det(11, b * nh * t_q * hd, 1.0);
    let k_f = det(22, b * nkv * t_kv * hd, 1.0);
    let v_f = det(33, b * nkv * t_kv * hd, 1.0);

    let q = Tensor::from_vec(to_bf16(&q_f), vec![b, nh, t_q, hd], dev).unwrap();
    let k = Tensor::from_vec(to_bf16(&k_f), vec![b, nkv, t_kv, hd], dev).unwrap();
    let v = Tensor::from_vec(to_bf16(&v_f), vec![b, nkv, t_kv, hd], dev).unwrap();

    let mut k_buf = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut v_buf = Tensor::zeros(vec![b, nkv, max_seq, hd], DType::MXFP8, dev).unwrap();
    let mut k_sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();
    let mut v_sc = Tensor::zeros(vec![b, nkv, max_seq, nb], DType::U8, dev).unwrap();

    k_buf.kv_append_quant_mxfp8_inplace(&mut k_sc, &k, 0).unwrap();
    v_buf.kv_append_quant_mxfp8_inplace(&mut v_sc, &v, 0).unwrap();

    let k_q = k_buf.narrow(2, 0, t_kv).unwrap();
    let v_q = v_buf.narrow(2, 0, t_kv).unwrap();
    let ks = k_sc.narrow(2, 0, t_kv).unwrap();
    let vs = v_sc.narrow(2, 0, t_kv).unwrap();

    let out = q.flash_attention_mxfp8kv(&k_q, &v_q, &ks, &vs, scale, true).unwrap();
    let out_f: Vec<f32> = out
        .to_dtype(DType::F32)
        .unwrap()
        .reshape(vec![b * nh * t_q * hd])
        .unwrap()
        .to_vec1()
        .unwrap();

    let q_b: Vec<f32> = to_bf16(&q_f).iter().map(|x| x.to_f32()).collect();
    let k_b: Vec<f32> = to_bf16(&k_f).iter().map(|x| x.to_f32()).collect();
    let v_b: Vec<f32> = to_bf16(&v_f).iter().map(|x| x.to_f32()).collect();
    let k_deq = quantize_roundtrip_block(&k_b, b, nkv, t_kv, hd);
    let v_deq = quantize_roundtrip_block(&v_b, b, nkv, t_kv, hd);
    let reference = cpu_sdpa(&q_b, &k_deq, &v_deq, b, nh, nkv, t_q, t_kv, hd, scale, true);

    let cs = cos_sim(&out_f, &reference);
    let max_abs = out_f.iter().zip(&reference).map(|(a, c)| (a - c).abs()).fold(0.0f32, f32::max);
    eprintln!("[mxfp8 flash {label}] cos={cs:.6} max_abs={max_abs:.5}");
    assert!(cs > 0.999, "{label}: cos {cs} (max_abs {max_abs})");
}

#[test]
fn mxfp8_kv_decode_m1() {
    if !setup() {
        return;
    }
    // decode: один query-токен (Tq=1) аттендит к 40 KV-токенам.
    run(2, 4, 2, 1, 40, 128, "decode_t1_kv40");
}

#[test]
fn mxfp8_kv_decode_v2_qwen38_shape() {
    if !setup() {
        return;
    }
    // Форма Qwen3.8-27B: 24Q/4KV×256 → v2-ядро GROUP=6. Tq=1 (decode) и
    // Tq=2 (MTP-verify), длины вокруг границ тайла (128) и сплитов.
    run_ms(1, 24, 4, 1, 40, 256, 64, "v2_g6_t1_kv40");
    run_ms(1, 24, 4, 1, 127, 256, 160, "v2_g6_t1_kv127");
    run_ms(1, 24, 4, 1, 128, 256, 160, "v2_g6_t1_kv128");
    run_ms(1, 24, 4, 1, 129, 256, 160, "v2_g6_t1_kv129");
    run_ms(1, 24, 4, 2, 700, 256, 704, "v2_g6_t2_kv700");
    run_ms(1, 24, 4, 8, 1100, 256, 1152, "v2_g6_t8_kv1100");
}

#[test]
fn mxfp8_kv_decode_v2_groups() {
    if !setup() {
        return;
    }
    // GROUP=1 (n_rep=1), GROUP=4 (16/4), GROUP=2 при hd=256, батч b=2.
    run_ms(1, 4, 4, 1, 300, 256, 320, "v2_g1_t1_kv300");
    run_ms(1, 16, 4, 1, 300, 256, 320, "v2_g4_t1_kv300");
    run_ms(2, 8, 4, 2, 200, 256, 256, "v2_g2_b2_t2_kv200");
    // n_rep=3 → GROUP=1, три сабгруппы.
    run_ms(1, 12, 4, 1, 200, 128, 256, "v2_nrep3_t1_kv200");
}

#[test]
fn mxfp8_kv_prefill_tensorcore() {
    if !setup() {
        return;
    }
    // prefill hd=128 Tq>1 → tensor-core flash_mxfp8_prefill. Малый Tq (q_count<BM).
    run(1, 4, 2, 12, 12, 128, "prefill_v4_t12");
    // Крупный prefill: multi Q-тайл + multi KV-блок.
    run(2, 8, 2, 40, 40, 128, "prefill_v4_t40");
}

#[test]
fn mxfp8_kv_prefill_hd64_scalar() {
    if !setup() {
        return;
    }
    // hd=64 (∉{128,256}) → scalar decode-путь даже при Tq>1.
    run(1, 4, 2, 12, 12, 64, "prefill_hd64_scalar");
}

#![cfg(feature = "cuda")]

use half::bf16;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use synaptix_kernels_cuda::attention::flash_bf16::FlashAttnBf16Kernels;

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_bf16(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            bf16::from_f32(f * scale)
        })
        .collect()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0_f64;
    let mut na = 0.0_f64;
    let mut nb = 0.0_f64;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

fn cpu_sdpa_bf16(
    q: &[bf16],
    k: &[bf16],
    v: &[bf16],
    b: usize,
    nh: usize,
    nkv: usize,
    t_q: usize,
    t_kv: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Vec<bf16> {
    let n_rep = nh / nkv;
    let mut out = vec![bf16::ZERO; b * nh * t_q * d];
    for bi in 0..b {
        for h in 0..nh {
            let h_kv = h / n_rep;
            for ti in 0..t_q {
                let q_pos_in_kv = if t_kv >= t_q { t_kv - t_q + ti } else { ti };
                let mut scores = vec![0.0_f32; t_kv];
                for j in 0..t_kv {
                    if causal && j > q_pos_in_kv {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let q_off = ((bi * nh + h) * t_q + ti) * d;
                    let k_off = ((bi * nkv + h_kv) * t_kv + j) * d;
                    let mut s = 0.0_f32;
                    for kk in 0..d {
                        s += q[q_off + kk].to_f32() * k[k_off + kk].to_f32();
                    }
                    scores[j] = s * scale;
                }
                let m = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut l = 0.0_f32;
                let mut e = vec![0.0_f32; t_kv];
                for j in 0..t_kv {
                    if scores[j].is_finite() {
                        e[j] = (scores[j] - m).exp();
                        l += e[j];
                    }
                }
                for kk in 0..d {
                    let mut acc = 0.0_f32;
                    for j in 0..t_kv {
                        if e[j] > 0.0 {
                            let v_off = ((bi * nkv + h_kv) * t_kv + j) * d;
                            acc += e[j] * v[v_off + kk].to_f32();
                        }
                    }
                    let out_off = ((bi * nh + h) * t_q + ti) * d;
                    out[out_off + kk] = bf16::from_f32(if l > 0.0 { acc / l } else { 0.0 });
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_bf16(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    b: usize,
    nh: usize,
    nkv: usize,
    t_q: usize,
    t_kv: usize,
    d: usize,
    causal: bool,
    name: &str,
    cos_floor: f32,
) {
    assert!(d % 128 == 0 && d <= 512);
    let scale = (d as f32).sqrt().recip();
    let q_host = det_bf16(0xA110_C8E1, b * nh * t_q * d, 0.3);
    let k_host = det_bf16(0xC0DE_BA5E, b * nkv * t_kv * d, 0.3);
    let v_host = det_bf16(0xDEAD_BEEF, b * nkv * t_kv * d, 0.3);

    let kernels = FlashAttnBf16Kernels::for_context(ctx).expect("compile flash_bf16");
    let dev_q: CudaSlice<bf16> = stream.clone_htod(&q_host).unwrap();
    let dev_k: CudaSlice<bf16> = stream.clone_htod(&k_host).unwrap();
    let dev_v: CudaSlice<bf16> = stream.clone_htod(&v_host).unwrap();
    let mut dev_out: CudaSlice<bf16> = stream.alloc_zeros(b * nh * t_q * d).unwrap();

    let n_rep = nh / nkv;
    let q_pos_base = if t_kv >= t_q { (t_kv - t_q) as u32 } else { 0 };
    kernels
        .flash_attn2_fwd(
            stream,
            &dev_q,
            &dev_k,
            &dev_v,
            &mut dev_out,
            scale,
            b as u32,
            nh as u32,
            nkv as u32,
            t_q as u32,
            t_kv as u32,
            d as u32,
            n_rep as u32,
            q_pos_base,
            if causal { 1 } else { 0 },
            t_kv as u32,
        )
        .expect("flash bf16 fwd");
    stream.synchronize().unwrap();

    let got_b: Vec<bf16> = stream.clone_dtoh(&dev_out).unwrap();
    let ref_b = cpu_sdpa_bf16(
        &q_host, &k_host, &v_host, b, nh, nkv, t_q, t_kv, d, scale, causal,
    );
    let got_f32: Vec<f32> = got_b.iter().map(|v| v.to_f32()).collect();
    let ref_f32: Vec<f32> = ref_b.iter().map(|v| v.to_f32()).collect();
    let cos = cos_sim(&got_f32, &ref_f32);
    eprintln!("[{name}] cos={cos:.6}");
    assert!(cos >= cos_floor, "{name}: cos={cos} < {cos_floor}");
}

#[test]
fn fa2_bf16_single_row_mha() {
    let Some((ctx, stream)) = setup() else { return };
    run_bf16(
        &ctx,
        &stream,
        1,
        4,
        4,
        1,
        16,
        128,
        false,
        "decode_mha",
        0.95,
    );
}

#[test]
fn fa2_bf16_single_row_gqa_causal() {
    let Some((ctx, stream)) = setup() else { return };
    run_bf16(
        &ctx,
        &stream,
        1,
        8,
        2,
        1,
        32,
        128,
        false,
        "decode_gqa",
        0.95,
    );
}

#[test]
fn fa2_bf16_tiled_prefill_d128() {
    let Some((ctx, stream)) = setup() else { return };
    run_bf16(
        &ctx,
        &stream,
        1,
        4,
        4,
        16,
        16,
        128,
        true,
        "prefill_tiled_d128_causal",
        0.95,
    );
}

#[test]
fn fa2_bf16_tiled_prefill_d256() {
    let Some((ctx, stream)) = setup() else { return };
    run_bf16(
        &ctx,
        &stream,
        1,
        2,
        2,
        8,
        16,
        256,
        false,
        "prefill_tiled_d256",
        0.95,
    );
}

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use synaptix_kernels_cuda::best_cu::gemv::gemv_nvfp4::{nvfp4_w_repack, Nvfp4MmaGemvShufKernels};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_dequant_f16, nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};
use synaptix_kernels_cuda::elementwise::rope::RopeKernels;
use synaptix_kernels_cuda::fused::attn_rope::fused_qkv_rope_f16;
use synaptix_kernels_cuda::fused::qkv_proj::Nvfp4QkvProjShufKernels;

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            let f = (u as f32 / u32::MAX as f32) * 2.0 - 1.0;
            f16::from_f32(f * scale)
        })
        .collect()
}

fn build_rope_tables(max_pos: usize, rotary_dim: usize, base: f32) -> (Vec<f16>, Vec<f16>) {
    let half = rotary_dim / 2;
    let mut cos = vec![f16::ZERO; max_pos * rotary_dim];
    let mut sin = vec![f16::ZERO; max_pos * rotary_dim];
    for p in 0..max_pos {
        for d in 0..half {
            let theta = (p as f32) * base.powf(-2.0 * (d as f32) / (rotary_dim as f32));
            let c = theta.cos();
            let s = theta.sin();
            cos[p * rotary_dim + d] = f16::from_f32(c);
            sin[p * rotary_dim + d] = f16::from_f32(s);
            cos[p * rotary_dim + d + half] = f16::from_f32(c);
            sin[p * rotary_dim + d + half] = f16::from_f32(s);
        }
    }
    (cos, sin)
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

fn apply_rope_ref(
    x: &[f32],
    cos: &[f16],
    sin: &[f16],
    h: usize,
    head_dim: usize,
    rotary_dim: usize,
    pos: usize,
) -> Vec<f32> {
    let half = rotary_dim / 2;
    let mut out = vec![0.0_f32; h * head_dim];
    for head in 0..h {
        for d in 0..head_dim {
            let i = head * head_dim + d;
            if d >= rotary_dim {
                out[i] = x[i];
                continue;
            }
            let c = cos[pos * rotary_dim + d].to_f32();
            let s = sin[pos * rotary_dim + d].to_f32();
            let v = x[i];
            let (partner, rot_sign) = if d < half {
                (x[head * head_dim + d + half], -1.0_f32)
            } else {
                (x[head * head_dim + d - half], 1.0_f32)
            };
            out[i] = v * c + rot_sign * partner * s;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    h_q: u32,
    h_kv: u32,
    head_dim: u32,
    k: u32,
    rotary_dim: u32,
) {
    let q = Nvfp4QuantKernels::for_context(ctx).expect("quant");
    let mma = Nvfp4MmaGemvShufKernels::for_context(ctx).expect("shuf");
    let qkv = Nvfp4QkvProjShufKernels::for_context(ctx).expect("qkv");
    let rope = RopeKernels::for_context(ctx).expect("rope");

    let n_q = h_q * head_dim;
    let n_kv = h_kv * head_dim;
    let pos: u32 = 7;

    let w_q_h = det_f16(0x11, (n_q * k) as usize, 0.3);
    let w_k_h = det_f16(0x22, (n_kv * k) as usize, 0.3);
    let w_v_h = det_f16(0x33, (n_kv * k) as usize, 0.3);
    let x_h = det_f16(0x44, k as usize, 0.3);

    let dev_w_q: CudaSlice<f16> = stream.clone_htod(&w_q_h).unwrap();
    let dev_w_k: CudaSlice<f16> = stream.clone_htod(&w_k_h).unwrap();
    let dev_w_v: CudaSlice<f16> = stream.clone_htod(&w_v_h).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_h).unwrap();

    let sf_w_q = nvfp4_scale_buffer_size(n_q as usize, k as usize);
    let sf_w_k = nvfp4_scale_buffer_size(n_kv as usize, k as usize);
    let sf_w_v = nvfp4_scale_buffer_size(n_kv as usize, k as usize);
    let sf_x = nvfp4_scale_buffer_size(1, k as usize);

    let mut w_q_p: CudaSlice<u8> = stream.alloc_zeros((n_q * k / 2) as usize).unwrap();
    let mut w_q_s: CudaSlice<u8> = stream.alloc_zeros(sf_w_q).unwrap();
    let mut w_k_p: CudaSlice<u8> = stream.alloc_zeros((n_kv * k / 2) as usize).unwrap();
    let mut w_k_s: CudaSlice<u8> = stream.alloc_zeros(sf_w_k).unwrap();
    let mut w_v_p: CudaSlice<u8> = stream.alloc_zeros((n_kv * k / 2) as usize).unwrap();
    let mut w_v_s: CudaSlice<u8> = stream.alloc_zeros(sf_w_v).unwrap();
    let mut x_p: CudaSlice<u8> = stream.alloc_zeros((k / 2) as usize).unwrap();
    let mut x_s: CudaSlice<u8> = stream.alloc_zeros(sf_x).unwrap();

    quantize_f16_to_nvfp4(&q, stream, &dev_w_q, &mut w_q_p, &mut w_q_s, n_q, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_w_k, &mut w_k_p, &mut w_k_s, n_kv, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_w_v, &mut w_v_p, &mut w_v_s, n_kv, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_x, &mut x_p, &mut x_s, 1, k).unwrap();

    let mut w_q_sh: CudaSlice<u8> = stream.alloc_zeros((n_q * k / 2) as usize).unwrap();
    let mut w_k_sh: CudaSlice<u8> = stream.alloc_zeros((n_kv * k / 2) as usize).unwrap();
    let mut w_v_sh: CudaSlice<u8> = stream.alloc_zeros((n_kv * k / 2) as usize).unwrap();
    nvfp4_w_repack(&mma, stream, &w_q_p, &mut w_q_sh, n_q, k).unwrap();
    nvfp4_w_repack(&mma, stream, &w_k_p, &mut w_k_sh, n_kv, k).unwrap();
    nvfp4_w_repack(&mma, stream, &w_v_p, &mut w_v_sh, n_kv, k).unwrap();

    let max_pos = (pos as usize) + 4;
    let (cos_h, sin_h) = build_rope_tables(max_pos, rotary_dim as usize, 10000.0);
    let dev_cos: CudaSlice<f16> = stream.clone_htod(&cos_h).unwrap();
    let dev_sin: CudaSlice<f16> = stream.clone_htod(&sin_h).unwrap();
    let dev_pos: CudaSlice<u32> = stream.clone_htod(&[pos]).unwrap();

    let mut out_q_proj: CudaSlice<f16> = stream.alloc_zeros(n_q as usize).unwrap();
    let mut out_q_roped: CudaSlice<f16> = stream.alloc_zeros(n_q as usize).unwrap();
    let mut out_k_proj: CudaSlice<f16> = stream.alloc_zeros(n_kv as usize).unwrap();
    let mut out_k_roped: CudaSlice<f16> = stream.alloc_zeros(n_kv as usize).unwrap();
    let mut out_v: CudaSlice<f16> = stream.alloc_zeros(n_kv as usize).unwrap();

    fused_qkv_rope_f16(
        &qkv,
        &rope,
        stream,
        &w_q_sh,
        &w_q_s,
        &w_k_sh,
        &w_k_s,
        &w_v_sh,
        &w_v_s,
        &x_p,
        &x_s,
        &mut out_q_proj,
        &mut out_q_roped,
        &mut out_k_proj,
        &mut out_k_roped,
        &mut out_v,
        &dev_cos,
        &dev_sin,
        &dev_pos,
        h_q,
        h_kv,
        head_dim,
        rotary_dim,
        k,
    )
    .expect("fused qkv+rope");

    let mut w_q_deq: CudaSlice<f16> = stream.alloc_zeros((n_q * k) as usize).unwrap();
    let mut w_k_deq: CudaSlice<f16> = stream.alloc_zeros((n_kv * k) as usize).unwrap();
    let mut w_v_deq: CudaSlice<f16> = stream.alloc_zeros((n_kv * k) as usize).unwrap();
    let mut x_deq: CudaSlice<f16> = stream.alloc_zeros(k as usize).unwrap();
    nvfp4_dequant_f16(&q, stream, &w_q_p, &w_q_s, &mut w_q_deq, n_q, k).unwrap();
    nvfp4_dequant_f16(&q, stream, &w_k_p, &w_k_s, &mut w_k_deq, n_kv, k).unwrap();
    nvfp4_dequant_f16(&q, stream, &w_v_p, &w_v_s, &mut w_v_deq, n_kv, k).unwrap();
    nvfp4_dequant_f16(&q, stream, &x_p, &x_s, &mut x_deq, 1, k).unwrap();
    stream.synchronize().unwrap();

    let w_q_deq_h: Vec<f16> = stream.clone_dtoh(&w_q_deq).unwrap();
    let w_k_deq_h: Vec<f16> = stream.clone_dtoh(&w_k_deq).unwrap();
    let w_v_deq_h: Vec<f16> = stream.clone_dtoh(&w_v_deq).unwrap();
    let x_deq_h: Vec<f16> = stream.clone_dtoh(&x_deq).unwrap();

    let mut q_proj_ref = vec![0.0_f32; n_q as usize];
    let mut k_proj_ref = vec![0.0_f32; n_kv as usize];
    let mut v_ref = vec![0.0_f32; n_kv as usize];
    for o in 0..(n_q as usize) {
        let mut acc = 0.0_f32;
        for j in 0..(k as usize) {
            acc += w_q_deq_h[o * (k as usize) + j].to_f32() * x_deq_h[j].to_f32();
        }
        q_proj_ref[o] = acc;
    }
    for o in 0..(n_kv as usize) {
        let mut ak = 0.0_f32;
        let mut av = 0.0_f32;
        for j in 0..(k as usize) {
            ak += w_k_deq_h[o * (k as usize) + j].to_f32() * x_deq_h[j].to_f32();
            av += w_v_deq_h[o * (k as usize) + j].to_f32() * x_deq_h[j].to_f32();
        }
        k_proj_ref[o] = ak;
        v_ref[o] = av;
    }

    let q_ref = apply_rope_ref(
        &q_proj_ref,
        &cos_h,
        &sin_h,
        h_q as usize,
        head_dim as usize,
        rotary_dim as usize,
        pos as usize,
    );
    let k_ref = apply_rope_ref(
        &k_proj_ref,
        &cos_h,
        &sin_h,
        h_kv as usize,
        head_dim as usize,
        rotary_dim as usize,
        pos as usize,
    );

    let out_q_h: Vec<f16> = stream.clone_dtoh(&out_q_roped).unwrap();
    let out_k_h: Vec<f16> = stream.clone_dtoh(&out_k_roped).unwrap();
    let out_v_h: Vec<f16> = stream.clone_dtoh(&out_v).unwrap();

    let out_q_f32: Vec<f32> = out_q_h.iter().map(|v| v.to_f32()).collect();
    let out_k_f32: Vec<f32> = out_k_h.iter().map(|v| v.to_f32()).collect();
    let out_v_f32: Vec<f32> = out_v_h.iter().map(|v| v.to_f32()).collect();

    let cos_q = cos_sim(&out_q_f32, &q_ref);
    let cos_k = cos_sim(&out_k_f32, &k_ref);
    let cos_v = cos_sim(&out_v_f32, &v_ref);
    eprintln!("[fused_attn_rope h_q={h_q} h_kv={h_kv} d={head_dim} K={k}] cos Q={cos_q:.5} K={cos_k:.5} V={cos_v:.5}");
    assert!(cos_q >= 0.99, "Q cos {cos_q} < 0.99");
    assert!(cos_k >= 0.99, "K cos {cos_k} < 0.99");
    assert!(cos_v >= 0.99, "V cos {cos_v} < 0.99");
}

#[test]
fn attn_rope_small_full_rotary() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 2, 1, 64, 128, 64);
}

#[test]
fn attn_rope_qwen3_1p7b_like() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 16, 8, 128, 2048, 128);
}

#[test]
fn attn_rope_partial_rotary() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 4, 2, 128, 1024, 64);
}

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;
use rayon::prelude::*;

use synaptix_kernels_cuda::best_cu::gemv::gemv_nvfp4::{
    nvfp4_mma_gemv_shuf_f16, nvfp4_w_repack, Nvfp4MmaGemvShufKernels,
};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_dequant_f16, nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};

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

fn run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    n: u32,
    k: u32,
    name: &str,
    cos_floor: f32,
) {
    let q = Nvfp4QuantKernels::for_context(ctx).expect("compile nvfp4_quant");
    let mma = Nvfp4MmaGemvShufKernels::for_context(ctx).expect("compile shuf");

    let w_host = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, k as usize, 0.5);

    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let w_scale_bytes = nvfp4_scale_buffer_size(n as usize, k as usize);
    let x_scale_bytes = nvfp4_scale_buffer_size(1, k as usize);

    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_scales: CudaSlice<u8> = stream.alloc_zeros(w_scale_bytes).unwrap();
    let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((k / 2) as usize).unwrap();
    let mut x_scales: CudaSlice<u8> = stream.alloc_zeros(x_scale_bytes).unwrap();

    quantize_f16_to_nvfp4(&q, stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_x, &mut x_packed, &mut x_scales, 1, k).unwrap();

    let mut w_packed_shuf: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    nvfp4_w_repack(&mma, stream, &w_packed, &mut w_packed_shuf, n, k).expect("repack");

    let mut y_ours: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();
    nvfp4_mma_gemv_shuf_f16(
        &mma,
        stream,
        &w_packed_shuf,
        &w_scales,
        &x_packed,
        &x_scales,
        &mut y_ours,
        n,
        k,
    )
    .expect("shuf mma gemv");

    let mut w_deq: CudaSlice<f16> = stream.alloc_zeros((n * k) as usize).unwrap();
    let mut x_deq: CudaSlice<f16> = stream.alloc_zeros(k as usize).unwrap();
    nvfp4_dequant_f16(&q, stream, &w_packed, &w_scales, &mut w_deq, n, k).unwrap();
    nvfp4_dequant_f16(&q, stream, &x_packed, &x_scales, &mut x_deq, 1, k).unwrap();
    stream.synchronize().unwrap();

    let w_deq_host: Vec<f16> = stream.clone_dtoh(&w_deq).unwrap();
    let x_deq_host: Vec<f16> = stream.clone_dtoh(&x_deq).unwrap();
    let y_ours_host: Vec<f16> = stream.clone_dtoh(&y_ours).unwrap();

    // GEMV-эталон на Qwen shapes (lm_head: N=248320, K=5120 ≈ 1.3 млрд MAC)
    // в один поток — десятки секунд. Пред-конвертим в f32, параллелим по выходу.
    let k_us = k as usize;
    let w_f32: Vec<f32> = w_deq_host.iter().map(|v| v.to_f32()).collect();
    let x_f32: Vec<f32> = x_deq_host.iter().map(|v| v.to_f32()).collect();
    let mut y_ref = vec![0.0_f32; n as usize];
    y_ref.par_iter_mut().enumerate().for_each(|(o, slot)| {
        let w_row = &w_f32[o * k_us..(o + 1) * k_us];
        let mut acc = 0.0_f32;
        for j in 0..k_us {
            acc += w_row[j] * x_f32[j];
        }
        *slot = acc;
    });
    let y_ours_f32: Vec<f32> = y_ours_host.iter().map(|v| v.to_f32()).collect();

    let cos_ref = cos_sim(&y_ours_f32, &y_ref);
    let mut max_abs_ref = 0.0_f32;
    for i in 0..(n as usize) {
        let d_ref = (y_ours_f32[i] - y_ref[i]).abs();
        if d_ref > max_abs_ref {
            max_abs_ref = d_ref;
        }
    }
    eprintln!(
        "[{name} shuf N={n} K={k}] vs CPU: cos={cos_ref:.6} max_abs={max_abs_ref:.4}"
    );
    assert!(
        cos_ref >= cos_floor,
        "{name}: cos vs CPU={cos_ref} < {cos_floor}"
    );
}

#[test]
fn shuf_64x64() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 64, 64, "64x64", 0.99);
}

#[test]
fn shuf_256x256() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 256, 256, "256x256", 0.99);
}

#[test]
fn shuf_5120x5120() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 5120, 5120, "qkv", 0.99);
}

#[test]
fn shuf_27648x5120() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 27648, 5120, "ffn_gate", 0.99);
}

#[test]
fn shuf_5120x27648() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 5120, 27648, "ffn_down", 0.99);
}

#[test]
fn shuf_lm_head_248320x5120() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 248320, 5120, "lm_head", 0.99);
}

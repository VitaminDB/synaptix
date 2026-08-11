
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::{bf16, f16};
use rayon::prelude::*;

use synaptix_kernels_cuda::best_cu::gemv::mma_gemv::{gemv_bf16, gemv_f16, gemv_f32, MmaGemvKernels};

fn setup() -> Option<(Arc<CudaContext>, Arc<CudaStream>)> {
    let ctx = synaptix_core::device::cuda::get(0).ok()?;
    let stream = synaptix_core::device::cuda::default_stream(0).ok()?;
    Some((ctx, stream))
}

fn det_seed(seed: u64, n: usize) -> Vec<f32> {
    let mut x = seed.wrapping_add(0x9E3779B97F4A7C15);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (x >> 33) as u32;
            (u as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn det_f16(seed: u64, n: usize, scale: f32) -> Vec<f16> {
    det_seed(seed, n)
        .into_iter()
        .map(|f| f16::from_f32(f * scale))
        .collect()
}
fn det_bf16(seed: u64, n: usize, scale: f32) -> Vec<bf16> {
    det_seed(seed, n)
        .into_iter()
        .map(|f| bf16::from_f32(f * scale))
        .collect()
}
fn det_f32(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    det_seed(seed, n).into_iter().map(|f| f * scale).collect()
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

// ─── F16 GEMV ───

#[test]
fn gemv_f16_qwen_attn_qkv() {
    let Some((ctx, stream)) = setup() else { return };
    let n = 5120u32;
    let k = 5120u32;
    let kernels = MmaGemvKernels::for_context(&ctx).expect("compile");

    let w = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x = det_f16(0xC0DE_BA5E, k as usize, 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x).unwrap();

    let mut y: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();
    gemv_f16(&kernels, &stream, &dev_w, &dev_x, &mut y, n, k).unwrap();
    stream.synchronize().unwrap();

    let y_h: Vec<f16> = stream.clone_dtoh(&y).unwrap();
    let y_f32: Vec<f32> = y_h.iter().map(|v| v.to_f32()).collect();

    // CPU ref через rayon par_iter_mut по выходным элементам.
    let k_us = k as usize;
    let w_f32: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let mut y_ref = vec![0.0_f32; n as usize];
    y_ref.par_iter_mut().enumerate().for_each(|(o, slot)| {
        let w_row = &w_f32[o * k_us..(o + 1) * k_us];
        let mut acc = 0.0_f32;
        for j in 0..k_us {
            acc += w_row[j] * x_f32[j];
        }
        *slot = acc;
    });
    let cos_cpu = cos_sim(&y_f32, &y_ref);
    eprintln!("[f16 attn_qkv GEMV] vs CPU: cos={cos_cpu:.6}");
    assert!(cos_cpu >= 0.99, "cos vs CPU={cos_cpu}");
}

// ─── BF16 GEMV ───

#[test]
fn gemv_bf16_qwen_attn_qkv() {
    let Some((ctx, stream)) = setup() else { return };
    let n = 5120u32;
    let k = 5120u32;
    let kernels = MmaGemvKernels::for_context(&ctx).expect("compile");

    let w = det_bf16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x = det_bf16(0xC0DE_BA5E, k as usize, 0.5);
    let dev_w: CudaSlice<bf16> = stream.clone_htod(&w).unwrap();
    let dev_x: CudaSlice<bf16> = stream.clone_htod(&x).unwrap();

    let mut y: CudaSlice<bf16> = stream.alloc_zeros(n as usize).unwrap();
    gemv_bf16(&kernels, &stream, &dev_w, &dev_x, &mut y, n, k).unwrap();
    stream.synchronize().unwrap();

    let y_h: Vec<bf16> = stream.clone_dtoh(&y).unwrap();
    let y_f32: Vec<f32> = y_h.iter().map(|v| v.to_f32()).collect();

    let k_us = k as usize;
    let w_f32: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
    let x_f32: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
    let mut y_ref = vec![0.0_f32; n as usize];
    y_ref.par_iter_mut().enumerate().for_each(|(o, slot)| {
        let w_row = &w_f32[o * k_us..(o + 1) * k_us];
        let mut acc = 0.0_f32;
        for j in 0..k_us {
            acc += w_row[j] * x_f32[j];
        }
        *slot = acc;
    });
    let cos_cpu = cos_sim(&y_f32, &y_ref);
    eprintln!("[bf16 attn_qkv GEMV] vs CPU: cos={cos_cpu:.6}");
    assert!(cos_cpu >= 0.99, "cos vs CPU={cos_cpu}");
}

// ─── F32 GEMV ───

#[test]
fn gemv_f32_qwen_attn_qkv() {
    let Some((ctx, stream)) = setup() else { return };
    let n = 5120u32;
    let k = 5120u32;
    let kernels = MmaGemvKernels::for_context(&ctx).expect("compile");

    let w = det_f32(0xA110_C8E1, (n * k) as usize, 0.5);
    let x = det_f32(0xC0DE_BA5E, k as usize, 0.5);
    let dev_w: CudaSlice<f32> = stream.clone_htod(&w).unwrap();
    let dev_x: CudaSlice<f32> = stream.clone_htod(&x).unwrap();

    let mut y: CudaSlice<f32> = stream.alloc_zeros(n as usize).unwrap();
    gemv_f32(&kernels, &stream, &dev_w, &dev_x, &mut y, n, k).unwrap();
    stream.synchronize().unwrap();

    let y_h: Vec<f32> = stream.clone_dtoh(&y).unwrap();
    let k_us = k as usize;
    let mut y_ref = vec![0.0_f32; n as usize];
    y_ref.par_iter_mut().enumerate().for_each(|(o, slot)| {
        let w_row = &w[o * k_us..(o + 1) * k_us];
        let mut acc = 0.0_f32;
        for j in 0..k_us {
            acc += w_row[j] * x[j];
        }
        *slot = acc;
    });
    let cos_cpu = cos_sim(&y_h, &y_ref);
    eprintln!("[f32 attn_qkv GEMV] vs CPU: cos={cos_cpu:.6}");
    assert!(cos_cpu >= 0.999, "cos vs CPU={cos_cpu}");
}

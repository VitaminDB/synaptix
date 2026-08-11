
use half::f16;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
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

fn run_roundtrip(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    outer: usize,
    inner: usize,
    name: &str,
    cos_floor: f32,
    max_abs_ceil: f32,
) {
    assert_eq!(inner % 16, 0, "{name}: inner must be multiple of 16");
    let x_host = det_f16(0xA110_C8E1, outer * inner, 0.7);
    let kernels = Nvfp4QuantKernels::for_context(ctx).expect("compile nvfp4_quant");

    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();
    let mut packed: CudaSlice<u8> = stream.alloc_zeros(outer * inner / 2).unwrap();
    let scale_bytes = nvfp4_scale_buffer_size(outer, inner);
    let mut scales: CudaSlice<u8> = stream.alloc_zeros(scale_bytes).unwrap();
    let mut dev_y: CudaSlice<f16> = stream.alloc_zeros(outer * inner).unwrap();

    quantize_f16_to_nvfp4(
        &kernels,
        stream,
        &dev_x,
        &mut packed,
        &mut scales,
        outer as u32,
        inner as u32,
    )
    .expect("quantize");
    nvfp4_dequant_f16(
        &kernels,
        stream,
        &packed,
        &scales,
        &mut dev_y,
        outer as u32,
        inner as u32,
    )
    .expect("dequant");
    stream.synchronize().unwrap();

    let y_host: Vec<f16> = stream.clone_dtoh(&dev_y).unwrap();

    let x_f32: Vec<f32> = x_host.iter().map(|v| v.to_f32()).collect();
    let y_f32: Vec<f32> = y_host.iter().map(|v| v.to_f32()).collect();

    let mut max_abs = 0.0_f32;
    let mut worst = (0usize, 0.0_f32, 0.0_f32);
    for i in 0..x_f32.len() {
        let d = (x_f32[i] - y_f32[i]).abs();
        if d > max_abs {
            max_abs = d;
            worst = (i, x_f32[i], y_f32[i]);
        }
    }
    let cos = cos_sim(&x_f32, &y_f32);
    eprintln!(
        "[{name} {}x{}] cos_sim={:.6}, max_abs={:.4} (worst i={} x={:.4} y={:.4})",
        outer, inner, cos, max_abs, worst.0, worst.1, worst.2
    );
    assert!(
        cos >= cos_floor,
        "{name}: cos_sim={cos} < floor {cos_floor}"
    );
    assert!(
        max_abs <= max_abs_ceil,
        "{name}: max_abs={max_abs} > ceil {max_abs_ceil}"
    );
}

#[test]
fn nvfp4_roundtrip_small_64x64() {
    let Some((ctx, stream)) = setup() else { return };
    run_roundtrip(&ctx, &stream, 64, 64, "small_64x64", 0.99, 0.15);
}

#[test]
fn nvfp4_roundtrip_64x128() {
    let Some((ctx, stream)) = setup() else { return };
    run_roundtrip(&ctx, &stream, 64, 128, "row64_k128", 0.99, 0.15);
}

#[test]
fn nvfp4_roundtrip_128x64() {
    let Some((ctx, stream)) = setup() else { return };
    run_roundtrip(&ctx, &stream, 128, 64, "row128_k64", 0.99, 0.15);
}

#[test]
fn nvfp4_roundtrip_multi_tile_256x128() {
    let Some((ctx, stream)) = setup() else { return };
    run_roundtrip(&ctx, &stream, 256, 128, "multi_tile_256x128", 0.99, 0.15);
}

#[test]
fn nvfp4_roundtrip_qwen_shape_5120x5120() {
    let Some((ctx, stream)) = setup() else { return };
    run_roundtrip(&ctx, &stream, 5120, 5120, "qwen_5120x5120", 0.99, 0.15);
}

#[test]
fn nvfp4_roundtrip_single_row_1xk() {
    let Some((ctx, stream)) = setup() else { return };
    run_roundtrip(&ctx, &stream, 1, 256, "act_1x256", 0.97, 0.15);
}

#[test]
fn nvfp4_roundtrip_idempotent() {
    let Some((ctx, stream)) = setup() else { return };
    let outer = 128usize;
    let inner = 128usize;
    let x_host = det_f16(0xDEAD_BEEF, outer * inner, 0.5);
    let kernels = Nvfp4QuantKernels::for_context(&ctx).expect("compile");

    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();
    let mut packed1: CudaSlice<u8> = stream.alloc_zeros(outer * inner / 2).unwrap();
    let mut packed2: CudaSlice<u8> = stream.alloc_zeros(outer * inner / 2).unwrap();
    let scale_bytes = nvfp4_scale_buffer_size(outer, inner);
    let mut scales1: CudaSlice<u8> = stream.alloc_zeros(scale_bytes).unwrap();
    let mut scales2: CudaSlice<u8> = stream.alloc_zeros(scale_bytes).unwrap();

    quantize_f16_to_nvfp4(
        &kernels,
        &stream,
        &dev_x,
        &mut packed1,
        &mut scales1,
        outer as u32,
        inner as u32,
    )
    .unwrap();
    quantize_f16_to_nvfp4(
        &kernels,
        &stream,
        &dev_x,
        &mut packed2,
        &mut scales2,
        outer as u32,
        inner as u32,
    )
    .unwrap();
    stream.synchronize().unwrap();

    let p1: Vec<u8> = stream.clone_dtoh(&packed1).unwrap();
    let p2: Vec<u8> = stream.clone_dtoh(&packed2).unwrap();
    let s1: Vec<u8> = stream.clone_dtoh(&scales1).unwrap();
    let s2: Vec<u8> = stream.clone_dtoh(&scales2).unwrap();
    assert_eq!(p1, p2, "quantize is not deterministic (packed)");
    assert_eq!(s1, s2, "quantize is not deterministic (scales)");
}

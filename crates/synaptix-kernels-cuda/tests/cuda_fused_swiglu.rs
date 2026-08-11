
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use synaptix_kernels_cuda::best_cu::gemv::gemv_nvfp4::{
    nvfp4_mma_gemv_shuf_f16, nvfp4_w_repack, Nvfp4MmaGemvShufKernels,
};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};
use synaptix_kernels_cuda::fused::swiglu::{nvfp4_swiglu_shuf_f16, Nvfp4SwigluShufKernels};

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

fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

fn run(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    n: u32,
    k: u32,
    name: &str,
    cos_floor: f32,
) {
    let q = Nvfp4QuantKernels::for_context(ctx).expect("quant");
    let mma = Nvfp4MmaGemvShufKernels::for_context(ctx).expect("mma");
    let glu = Nvfp4SwigluShufKernels::for_context(ctx).expect("swiglu");

    let w_g_h = det_f16(0x71, (n * k) as usize, 0.3);
    let w_u_h = det_f16(0x72, (n * k) as usize, 0.3);
    let x_h = det_f16(0x73, k as usize, 0.3);

    let dev_w_g: CudaSlice<f16> = stream.clone_htod(&w_g_h).unwrap();
    let dev_w_u: CudaSlice<f16> = stream.clone_htod(&w_u_h).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_h).unwrap();

    let sf_w = nvfp4_scale_buffer_size(n as usize, k as usize);
    let sf_x = nvfp4_scale_buffer_size(1, k as usize);

    let mut w_g_p: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_g_s: CudaSlice<u8> = stream.alloc_zeros(sf_w).unwrap();
    let mut w_u_p: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_u_s: CudaSlice<u8> = stream.alloc_zeros(sf_w).unwrap();
    let mut x_p: CudaSlice<u8> = stream.alloc_zeros((k / 2) as usize).unwrap();
    let mut x_s: CudaSlice<u8> = stream.alloc_zeros(sf_x).unwrap();

    quantize_f16_to_nvfp4(&q, stream, &dev_w_g, &mut w_g_p, &mut w_g_s, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_w_u, &mut w_u_p, &mut w_u_s, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_x, &mut x_p, &mut x_s, 1, k).unwrap();

    let mut w_g_sh: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_u_sh: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    nvfp4_w_repack(&mma, stream, &w_g_p, &mut w_g_sh, n, k).unwrap();
    nvfp4_w_repack(&mma, stream, &w_u_p, &mut w_u_sh, n, k).unwrap();

    let mut out: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();
    nvfp4_swiglu_shuf_f16(
        &glu, stream, &w_g_sh, &w_g_s, &w_u_sh, &w_u_s, &x_p, &x_s, &mut out, n, k,
    )
    .expect("fused swiglu");

    let mut g_only: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();
    let mut u_only: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();
    nvfp4_mma_gemv_shuf_f16(&mma, stream, &w_g_sh, &w_g_s, &x_p, &x_s, &mut g_only, n, k).unwrap();
    nvfp4_mma_gemv_shuf_f16(&mma, stream, &w_u_sh, &w_u_s, &x_p, &x_s, &mut u_only, n, k).unwrap();

    stream.synchronize().unwrap();

    let out_h: Vec<f16> = stream.clone_dtoh(&out).unwrap();
    let g_h: Vec<f16> = stream.clone_dtoh(&g_only).unwrap();
    let u_h: Vec<f16> = stream.clone_dtoh(&u_only).unwrap();

    let out_f32: Vec<f32> = out_h.iter().map(|v| v.to_f32()).collect();
    let ref_f32: Vec<f32> = (0..n as usize)
        .map(|i| silu(g_h[i].to_f32()) * u_h[i].to_f32())
        .collect();

    let cos = cos_sim(&out_f32, &ref_f32);
    let mut max_abs = 0.0_f32;
    for i in 0..n as usize {
        let d = (out_f32[i] - ref_f32[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let mut max_ref_abs = 0.0_f32;
    for &v in &ref_f32 {
        if v.abs() > max_ref_abs {
            max_ref_abs = v.abs();
        }
    }
    let rel = max_abs / max_ref_abs.max(1e-6);
    eprintln!("[swiglu {name} N={n} K={k}] cos={cos:.6} max_abs={max_abs:.4} (rel={rel:.4})");
    assert!(cos >= cos_floor, "{name}: cos={cos} < {cos_floor}");
    assert!(rel <= 0.01, "{name}: rel_max_abs={rel} > 0.01");
}

#[test]
fn swiglu_w4_small() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 256, 128, "small", 0.99);
}

#[test]
fn swiglu_w8_qwen3_1p7b_ffn() {
    // Qwen3 1.7B: hidden=2048, intermediate≈6144 — round to 6144 (×128).
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 6144, 2048, "qwen3_1p7b", 0.99);
}

#[test]
fn swiglu_w8_qwen3_27648x5120() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 27648, 5120, "qwen3_27648x5120", 0.99);
}

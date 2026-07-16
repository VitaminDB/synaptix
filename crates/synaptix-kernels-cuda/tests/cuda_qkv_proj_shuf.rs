#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use half::f16;

use synaptix_kernels_cuda::best_cu::gemv::gemv_nvfp4::{
    nvfp4_mma_gemv_shuf_f16, nvfp4_w_repack, Nvfp4MmaGemvShufKernels,
};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};
use synaptix_kernels_cuda::fused::qkv_proj::{nvfp4_qkv_proj_shuf_f16, Nvfp4QkvProjShufKernels};

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

fn run(ctx: &Arc<CudaContext>, stream: &Arc<CudaStream>, n_q: u32, n_k: u32, n_v: u32, k: u32) {
    let q = Nvfp4QuantKernels::for_context(ctx).expect("compile nvfp4_quant");
    let mma = Nvfp4MmaGemvShufKernels::for_context(ctx).expect("compile shuf");
    let qkv = Nvfp4QkvProjShufKernels::for_context(ctx).expect("compile qkv_proj");

    let w_q_host = det_f16(0xA001, (n_q * k) as usize, 0.4);
    let w_k_host = det_f16(0xA002, (n_k * k) as usize, 0.4);
    let w_v_host = det_f16(0xA003, (n_v * k) as usize, 0.4);
    let x_host = det_f16(0xB001, k as usize, 0.4);

    let dev_w_q: CudaSlice<f16> = stream.clone_htod(&w_q_host).unwrap();
    let dev_w_k: CudaSlice<f16> = stream.clone_htod(&w_k_host).unwrap();
    let dev_w_v: CudaSlice<f16> = stream.clone_htod(&w_v_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let sf_w_q = nvfp4_scale_buffer_size(n_q as usize, k as usize);
    let sf_w_k = nvfp4_scale_buffer_size(n_k as usize, k as usize);
    let sf_w_v = nvfp4_scale_buffer_size(n_v as usize, k as usize);
    let sf_x = nvfp4_scale_buffer_size(1, k as usize);

    let mut w_q_p: CudaSlice<u8> = stream.alloc_zeros((n_q * k / 2) as usize).unwrap();
    let mut w_q_s: CudaSlice<u8> = stream.alloc_zeros(sf_w_q).unwrap();
    let mut w_k_p: CudaSlice<u8> = stream.alloc_zeros((n_k * k / 2) as usize).unwrap();
    let mut w_k_s: CudaSlice<u8> = stream.alloc_zeros(sf_w_k).unwrap();
    let mut w_v_p: CudaSlice<u8> = stream.alloc_zeros((n_v * k / 2) as usize).unwrap();
    let mut w_v_s: CudaSlice<u8> = stream.alloc_zeros(sf_w_v).unwrap();
    let mut x_p: CudaSlice<u8> = stream.alloc_zeros((k / 2) as usize).unwrap();
    let mut x_s: CudaSlice<u8> = stream.alloc_zeros(sf_x).unwrap();

    quantize_f16_to_nvfp4(&q, stream, &dev_w_q, &mut w_q_p, &mut w_q_s, n_q, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_w_k, &mut w_k_p, &mut w_k_s, n_k, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_w_v, &mut w_v_p, &mut w_v_s, n_v, k).unwrap();
    quantize_f16_to_nvfp4(&q, stream, &dev_x, &mut x_p, &mut x_s, 1, k).unwrap();

    let mut w_q_sh: CudaSlice<u8> = stream.alloc_zeros((n_q * k / 2) as usize).unwrap();
    let mut w_k_sh: CudaSlice<u8> = stream.alloc_zeros((n_k * k / 2) as usize).unwrap();
    let mut w_v_sh: CudaSlice<u8> = stream.alloc_zeros((n_v * k / 2) as usize).unwrap();
    nvfp4_w_repack(&mma, stream, &w_q_p, &mut w_q_sh, n_q, k).expect("repack q");
    nvfp4_w_repack(&mma, stream, &w_k_p, &mut w_k_sh, n_k, k).expect("repack k");
    nvfp4_w_repack(&mma, stream, &w_v_p, &mut w_v_sh, n_v, k).expect("repack v");

    let mut out_q: CudaSlice<f16> = stream.alloc_zeros(n_q as usize).unwrap();
    let mut out_k: CudaSlice<f16> = stream.alloc_zeros(n_k as usize).unwrap();
    let mut out_v: CudaSlice<f16> = stream.alloc_zeros(n_v as usize).unwrap();
    nvfp4_qkv_proj_shuf_f16(
        &qkv, stream, &w_q_sh, &w_q_s, &w_k_sh, &w_k_s, &w_v_sh, &w_v_s, &x_p, &x_s, &mut out_q,
        &mut out_k, &mut out_v, n_q, n_k, n_v, k,
    )
    .expect("qkv fused");

    let mut ref_q: CudaSlice<f16> = stream.alloc_zeros(n_q as usize).unwrap();
    let mut ref_k: CudaSlice<f16> = stream.alloc_zeros(n_k as usize).unwrap();
    let mut ref_v: CudaSlice<f16> = stream.alloc_zeros(n_v as usize).unwrap();
    nvfp4_mma_gemv_shuf_f16(
        &mma, stream, &w_q_sh, &w_q_s, &x_p, &x_s, &mut ref_q, n_q, k,
    )
    .unwrap();
    nvfp4_mma_gemv_shuf_f16(
        &mma, stream, &w_k_sh, &w_k_s, &x_p, &x_s, &mut ref_k, n_k, k,
    )
    .unwrap();
    nvfp4_mma_gemv_shuf_f16(
        &mma, stream, &w_v_sh, &w_v_s, &x_p, &x_s, &mut ref_v, n_v, k,
    )
    .unwrap();

    stream.synchronize().unwrap();

    let out_q_h: Vec<f16> = stream.clone_dtoh(&out_q).unwrap();
    let out_k_h: Vec<f16> = stream.clone_dtoh(&out_k).unwrap();
    let out_v_h: Vec<f16> = stream.clone_dtoh(&out_v).unwrap();
    let ref_q_h: Vec<f16> = stream.clone_dtoh(&ref_q).unwrap();
    let ref_k_h: Vec<f16> = stream.clone_dtoh(&ref_k).unwrap();
    let ref_v_h: Vec<f16> = stream.clone_dtoh(&ref_v).unwrap();

    for (name, a, b) in [
        ("Q", &out_q_h, &ref_q_h),
        ("K", &out_k_h, &ref_k_h),
        ("V", &out_v_h, &ref_v_h),
    ] {
        assert_eq!(a.len(), b.len(), "{name}: length mismatch");
        for i in 0..a.len() {
            let av = a[i].to_f32();
            let bv = b[i].to_f32();
            assert!(
                (av - bv).abs() <= 1e-3,
                "{name}[{i}] mismatch: fused={av} ref={bv}"
            );
        }
    }
}

#[test]
fn qkv_w4_small() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 128, 64, 64, 128);
}

#[test]
fn qkv_w4_qwen3_1p7b() {
    // Qwen3 1.7B: heads_q=14×128=1792 (not /64), use 16×128=2048; KV=8×128=1024; K=2048
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 2048, 1024, 1024, 2048);
}

#[test]
fn qkv_w8_qwen3_4096() {
    let Some((ctx, stream)) = setup() else { return };
    run(&ctx, &stream, 4096, 1024, 1024, 4096);
}

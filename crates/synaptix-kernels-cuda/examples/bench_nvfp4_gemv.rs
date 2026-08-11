//! Микробенч NVFP4 weight-only GEMV (M=1 decode) на реальных формах Qwen3.6-27B.
//! Замеряет время/вызов и эффективную DRAM-bandwidth (вес читается раз/вызов).
//! NVRTC: правка gemv_nvfp4.cu подхватывается без пересборки Rust.
//! Запуск: cargo run --release --example bench_nvfp4_gemv --features cuda

use cudarc::driver::CudaSlice;
use half::f16;

use synaptix_kernels_cuda::best_cu::gemv::gemv_nvfp4::{
    nvfp4_mma_gemv_shuf_f16, nvfp4_w_repack, Nvfp4MmaGemvShufKernels,
};
use synaptix_kernels_cuda::elementwise::quant::{
    nvfp4_scale_buffer_size, quantize_f16_to_nvfp4, Nvfp4QuantKernels,
};

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

fn main() {
    let ctx = synaptix_core::device::cuda::get(0).expect("cuda ctx");
    let stream = synaptix_core::device::cuda::default_stream(0).expect("stream");
    let q = Nvfp4QuantKernels::for_context(&ctx).expect("nvfp4_quant");
    let mma = Nvfp4MmaGemvShufKernels::for_context(&ctx).expect("gemv");

    // (name, N, K) — формы decode-проекций Qwen3.6-27B hybrid.
    let shapes: &[(&str, u32, u32)] = &[
        ("mlp_gate/up  ", 17408, 5120),
        ("mlp_down     ", 5120, 17408),
        ("lin_in_qkv   ", 10240, 5120),
        ("lin_in_z     ", 6144, 5120),
        ("lin_oproj    ", 5120, 6144),
        ("attn_qproj   ", 12288, 5120),
        ("attn_kvproj  ", 1024, 5120),
        ("attn_oproj   ", 5120, 6144),
        ("lm_head      ", 248320, 5120),
    ];

    let peak_gbps = std::env::var("PEAK_GBPS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(896.0);
    let iters = 300u32;
    let warmup = 30u32;

    println!(
        "{:<14}{:>8}{:>8}{:>10}{:>12}{:>9}",
        "shape", "N", "K", "us/call", "GB/s", "%peak"
    );
    let mut total_us = 0.0f64;
    for &(name, n, k) in shapes {
        let w_host = det_f16(0xA110_C8E1, (n as usize) * (k as usize), 0.5);
        let x_host = det_f16(0xC0DE_BA5E, k as usize, 0.5);
        let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
        let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

        let mut w_packed: CudaSlice<u8> =
            stream.alloc_zeros((n as usize) * (k as usize) / 2).unwrap();
        let mut w_scales: CudaSlice<u8> = stream
            .alloc_zeros(nvfp4_scale_buffer_size(n as usize, k as usize))
            .unwrap();
        let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((k as usize) / 2).unwrap();
        let mut x_scales: CudaSlice<u8> = stream
            .alloc_zeros(nvfp4_scale_buffer_size(1, k as usize))
            .unwrap();
        quantize_f16_to_nvfp4(&q, &stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
        quantize_f16_to_nvfp4(&q, &stream, &dev_x, &mut x_packed, &mut x_scales, 1, k).unwrap();
        let wbytes_one = (n as usize) * (k as usize) / 2;
        // Ротация копий веса, чтобы рабочий набор ≫ L2 (~512MB) → cold-DRAM, как в
        // реальном decode (каждый слой читает свой вес раз/токен, L2-reuse нет).
        let target_ws: usize = 512 * 1024 * 1024;
        let copies = (target_ws / wbytes_one.max(1)).clamp(1, 24);
        // Для замера ВРЕМЕНИ значения копий не важны (DRAM-трафик одинаков), поэтому
        // repack только в [0], остальные — нули; ротация по разным буферам бьёт мимо L2.
        let mut w_copies: Vec<CudaSlice<u8>> = Vec::with_capacity(copies);
        let mut s_copies: Vec<CudaSlice<u8>> = Vec::with_capacity(copies);
        for c in 0..copies {
            let mut wc: CudaSlice<u8> = stream.alloc_zeros(wbytes_one).unwrap();
            if c == 0 {
                nvfp4_w_repack(&mma, &stream, &w_packed, &mut wc, n, k).unwrap();
            }
            let sc: CudaSlice<u8> =
                stream.alloc_zeros(nvfp4_scale_buffer_size(n as usize, k as usize)).unwrap();
            w_copies.push(wc);
            s_copies.push(sc);
        }
        let mut y: CudaSlice<f16> = stream.alloc_zeros(n as usize).unwrap();

        for i in 0..warmup {
            let c = (i as usize) % copies;
            nvfp4_mma_gemv_shuf_f16(&mma, &stream, &w_copies[c], &s_copies[c], &x_packed, &x_scales, &mut y, n, k).unwrap();
        }
        stream.synchronize().unwrap();
        let t0 = std::time::Instant::now();
        for i in 0..iters {
            let c = (i as usize) % copies;
            nvfp4_mma_gemv_shuf_f16(&mma, &stream, &w_copies[c], &s_copies[c], &x_packed, &x_scales, &mut y, n, k).unwrap();
        }
        stream.synchronize().unwrap();
        let us = t0.elapsed().as_secs_f64() * 1e6 / iters as f64;
        // Байты на вызов: NVFP4-вес (N*K/2) + E4M3 block-scale (N*K/16) + активация (K/2) + выход (N*2).
        let wbytes = (n as f64) * (k as f64) * (0.5 + 0.0625) + (k as f64) * 0.5 + (n as f64) * 2.0;
        let gbps = wbytes / (us * 1e3);
        total_us += us;
        println!(
            "{:<14}{:>8}{:>8}{:>10.2}{:>12.1}{:>8.1}%",
            name, n, k, us, gbps, 100.0 * gbps / peak_gbps
        );
    }
    println!("\nΣ us/call (по одному каждой формы) = {:.2}", total_us);
}

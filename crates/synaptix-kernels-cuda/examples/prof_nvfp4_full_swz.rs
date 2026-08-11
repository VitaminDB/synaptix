//! Focused ncu target: gn_nvfp4_full_128x128_c256_s4_swz on ffn_gate vs attn_qkv at M=1024.
//! Runs 10 warmup + 1 profiled launch of c256_s4_swz only.
//! Shape via env SHAPE=attn|gate (default attn). M via env M (default 1024).
//! ncu: `-k regex:c256_s4_swz -s 10 -c 1`.

use cudarc::driver::CudaSlice;
use half::f16;

use synaptix_kernels_cuda::best_cu::gemm::gemm_nvfp4::{
    gemm_nvfp4_full_cfg_view, GemmNvfp4FullKernels, Nvfp4FullCfg,
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
    let full = GemmNvfp4FullKernels::for_context(&ctx).expect("nvfp4_full");

    let shape = std::env::var("SHAPE").unwrap_or_else(|_| "attn".to_string());
    let batch: u32 = std::env::var("M")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);
    let (n, k) = match shape.as_str() {
        "gate" => (27648u32, 5120u32),
        _ => (5120u32, 5120u32),
    };

    let w_host = det_f16(0xA110_C8E1, (n * k) as usize, 0.5);
    let x_host = det_f16(0xC0DE_BA5E, (batch * k) as usize, 0.5);
    let dev_w: CudaSlice<f16> = stream.clone_htod(&w_host).unwrap();
    let dev_x: CudaSlice<f16> = stream.clone_htod(&x_host).unwrap();

    let mut w_packed: CudaSlice<u8> = stream.alloc_zeros((n * k / 2) as usize).unwrap();
    let mut w_scales: CudaSlice<u8> = stream
        .alloc_zeros(nvfp4_scale_buffer_size(n as usize, k as usize))
        .unwrap();
    let mut x_packed: CudaSlice<u8> = stream.alloc_zeros((batch * k / 2) as usize).unwrap();
    let mut x_scales: CudaSlice<u8> = stream
        .alloc_zeros(nvfp4_scale_buffer_size(batch as usize, k as usize))
        .unwrap();

    quantize_f16_to_nvfp4(&q, &stream, &dev_w, &mut w_packed, &mut w_scales, n, k).unwrap();
    quantize_f16_to_nvfp4(&q, &stream, &dev_x, &mut x_packed, &mut x_scales, batch, k).unwrap();

    let mut y: CudaSlice<f16> = stream.alloc_zeros((batch * n) as usize).unwrap();
    let cfg = Nvfp4FullCfg::C_128_128_C256_S4_SWZ;

    for _ in 0..11 {
        let mut yv = y.as_view_mut();
        gemm_nvfp4_full_cfg_view(
            &full, &stream, &w_packed, &w_scales, &x_packed, &x_scales, &mut yv, n, k, batch, cfg,
        )
        .unwrap();
    }
    stream.synchronize().unwrap();
    println!("done: c256_s4_swz shape={shape} {n}x{k} M={batch}");
}

//! Микробенч flash-attention на LTX stage2 FullHD-форме [1,32,32640,128] bf16
//! (v_attn1 = 64% шага по SYN_LTX_PROF). Замер ms/вызов + TFLOPS без QKV-GEMM
//! и транспозов. ncu: echo 1|sudo -S ncu -k regex:flash_splitq -c 3 -s 5 <bin>.

#![cfg(feature = "cuda")]

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

#[test]
fn flash_bench_ltx_stage2() {
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let (b, h, s, d) = (1usize, 32usize, 32640usize, 128usize);
    let n = b * h * s * d;
    let mk = || {
        let v: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.2).collect();
        Tensor::from_vec(v, (b, h, s, d), dev).unwrap().to_dtype(DType::BF16).unwrap()
    };
    let (q, k, v) = (mk(), mk(), mk());
    let scale = 1.0f32 / (d as f32).sqrt();

    for _ in 0..2 {
        let _ = q.flash_attention(&k, &v, scale, false).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let iters = 5;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = q.flash_attention(&k, &v, scale, false).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let flops = 2.0 * 2.0 * (b * h) as f64 * (s as f64) * (s as f64) * (d as f64);
    eprintln!("flash [1,32,32640,128] bf16: {ms:.1} ms/call, {:.1} TFLOPS", flops / (ms * 1e9));

    // та же форма stage1 (Tv 8160) для сравнения масштабирования по S
    let s1 = 8160usize;
    let n1 = b * h * s1 * d;
    let mk1 = || {
        let v: Vec<f32> = (0..n1).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.2).collect();
        Tensor::from_vec(v, (b, h, s1, d), dev).unwrap().to_dtype(DType::BF16).unwrap()
    };
    let (q1, k1, v1) = (mk1(), mk1(), mk1());
    for _ in 0..2 {
        let _ = q1.flash_attention(&k1, &v1, scale, false).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = q1.flash_attention(&k1, &v1, scale, false).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let ms1 = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    let flops1 = 2.0 * 2.0 * (b * h) as f64 * (s1 as f64) * (s1 as f64) * (d as f64);
    eprintln!("flash [1,32,8160,128] bf16: {ms1:.1} ms/call, {:.1} TFLOPS", flops1 / (ms1 * 1e9));
}

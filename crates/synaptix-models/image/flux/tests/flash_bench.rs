//! Микробенч flash-attention на FLUX-1024²-форме [1,24,4608,128] bf16 — крошечный
//! footprint (без 23GB-трансформера) → ncu профилируется быстро. Замер ms/вызов +
//! TFLOPS. ncu: echo 1|sudo -S ncu -k regex:flash_splitq_bf16_hd128 -c 3 -s 5 <bin>.


use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

#[test]
fn flash_bench_1024() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let dev = Device::Cuda(0);
    let (b, h, s, d) = (1usize, 24usize, 4608usize, 128usize);
    let n = b * h * s * d;
    let mk = || {
        let v: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 / 97.0 - 0.5) * 0.2).collect();
        Tensor::from_vec(v, (b, h, s, d), dev).unwrap().to_dtype(DType::BF16).unwrap()
    };
    let (q, k, v) = (mk(), mk(), mk());
    let scale = 1.0f32 / (d as f32).sqrt();

    // warmup
    for _ in 0..3 {
        let _ = q.flash_attention(&k, &v, scale, false).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let iters = 50;
    let t0 = std::time::Instant::now();
    for _ in 0..iters {
        let _ = q.flash_attention(&k, &v, scale, false).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    // FLOPs: QKᵀ + PV = 2 * 2 * b*h*s*s*d
    let flops = 2.0 * 2.0 * (b * h) as f64 * (s as f64) * (s as f64) * (d as f64);
    eprintln!("flash [1,24,4608,128] bf16: {ms:.3} ms/call, {:.1} TFLOPS", flops / (ms * 1e9));

    // транспозы: вход [B,S,H,D]→[B,H,S,D] (узкое) vs выход [B,H,S,D]→[B,S,H,D]
    let bshd = {
        let v: Vec<f32> = (0..n).map(|i| (i % 13) as f32 * 0.01).collect();
        Tensor::from_vec(v, (b, s, h, d), dev).unwrap().to_dtype(DType::BF16).unwrap()
    };
    let bhsd = q.clone(); // [b,h,s,d]
    let bench = |label: &str, t: &Tensor| {
        for _ in 0..3 { let _ = t.transpose(1, 2).unwrap().contiguous().unwrap(); }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let t0 = std::time::Instant::now();
        for _ in 0..iters { let _ = t.transpose(1, 2).unwrap().contiguous().unwrap(); }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let gbps = 2.0 * (n as f64) * 2.0 / (ms * 1e6); // read+write bf16
        eprintln!("transpose+contig {label} {:?}: {ms:.3} ms, {gbps:.0} GB/s", t.dims());
    };
    bench("INPUT  [B,S,H,D]→[B,H,S,D]", &bshd);
    bench("OUTPUT [B,H,S,D]→[B,S,H,D]", &bhsd);
}

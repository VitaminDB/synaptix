//! Бенч dense BF16 GEMM (Tensor::matmul → cutlass dense_gemm) на SDXL FF-формах.
//! Сравнение с torch.matmul (scripts/reference). TFLOP/s = 2*M*N*K / time.

use std::time::Instant;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn bench(m: usize, k: usize, n: usize, iters: usize) {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let dev = Device::Cuda(0);
    let a = Tensor::zeros(vec![m, k], DType::BF16, dev).unwrap();
    let b = Tensor::zeros(vec![k, n], DType::BF16, dev).unwrap();
    // warmup
    for _ in 0..5 {
        let _ = a.matmul(&b).unwrap();
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let t = Instant::now();
    for _ in 0..iters {
        let c = a.matmul(&b).unwrap();
        std::hint::black_box(&c);
    }
    synaptix_core::device::cuda::synchronize(0).unwrap();
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    let tflops = 2.0 * m as f64 * n as f64 * k as f64 / dt / 1e12;
    println!(
        "  M={m:5} K={k:5} N={n:6}: {:.3} ms  {tflops:6.1} TFLOP/s",
        dt * 1e3
    );
}

fn main() {
    println!("synaptix dense BF16 GEMM (cutlass):");
    // SDXL FF (dim 1280, inner 5120), batch2 × 32²=1024 → M=2048
    bench(2048, 1280, 10240, 30); // FF proj
    bench(2048, 5120, 1280, 30); // FF out
                                 // dim 640 (up1 64²=4096, batch2 → M=8192)
    bench(8192, 640, 5120, 20); // FF proj @ up1
    bench(8192, 2560, 640, 20); // FF out @ up1
                                // attention proj (down2: M=2048, dim 1280)
    bench(2048, 1280, 1280, 30);
    // квадрат для референса
    bench(4096, 4096, 4096, 20);
}

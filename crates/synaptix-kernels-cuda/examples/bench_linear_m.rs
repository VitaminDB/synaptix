//! Бенч BF16 `linear` (x·Wᵀ, TN) по M: неровные M (тайлы VAE H3 с
//! регистровыми токенами) против ровных. `bench_linear_m K N M…`

use std::time::Instant;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn main() {
    synaptix_kernels_cuda::cuda_backend::ensure_registered();
    let a: Vec<usize> = std::env::args().skip(1).map(|s| s.parse().unwrap()).collect();
    let (k, n) = (a[0], a[1]);
    let dev = Device::Cuda(0);
    let w = Tensor::ones(vec![n, k], DType::BF16, dev).unwrap();
    for &m in &a[2..] {
        let x = Tensor::ones(vec![m, k], DType::BF16, dev).unwrap();
        for _ in 0..5 {
            let _ = x.linear(&w).unwrap();
        }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let it = 50;
        let t = Instant::now();
        for _ in 0..it {
            std::hint::black_box(x.linear(&w).unwrap());
        }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let dt = t.elapsed().as_secs_f64() / it as f64;
        let tf = 2.0 * (m * n * k) as f64 / dt / 1e12;
        println!("M={m:5} K={k} N={n}: {:.3} мс  {tf:5.1} TFLOP/s", dt * 1e3);
    }
}

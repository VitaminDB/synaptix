//! Диагностика дыры bf16-стрима LTX: прод-путь dense linear (linear_bias_residual →
//! best_cu TN) на формах DiT-блока stage1/stage2 vs ожидание bf16-карты (~cuBLAS).
#![cfg(feature = "cuda")]

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

#[test]
fn ltx_bf16_dense_shapes() {
    if !setup() {
        return;
    }
    let _ng = synaptix_core::grad::NoGradGuard::new();
    let dev = Device::Cuda(0);
    for (m, n, k) in [
        (3520usize, 4096usize, 4096usize),
        (3584, 4096, 4096),
        (4096, 4096, 4096),
        (4992, 4096, 4096),
        (4992, 16384, 4096),
        (8192, 4096, 4096),
        (14080, 4096, 4096),
    ] {
        let dt = if std::env::var("SYN_DBG_F16").is_ok() { DType::F16 } else { DType::BF16 };
        let w = Tensor::randn(vec![n, k], Device::Cpu)
            .unwrap().to_device(dev).unwrap().mul_scalar(0.05).unwrap()
            .to_dtype(dt).unwrap();
        let b = Tensor::randn(vec![n], Device::Cpu)
            .unwrap().to_device(dev).unwrap().mul_scalar(0.01).unwrap()
            .to_dtype(dt).unwrap();
        let x = Tensor::randn(vec![1, m, k], Device::Cpu)
            .unwrap().to_device(dev).unwrap().mul_scalar(0.05).unwrap()
            .to_dtype(dt).unwrap();
        let iters: usize = std::env::var("SYN_DBG_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        for with_bias in [true, false] {
            let run = || {
                let y = x.linear_bias_residual(&w, if with_bias { Some(&b) } else { None }, None).unwrap();
                std::hint::black_box(&y);
            };
            for _ in 0..5 { run(); }
            synaptix_core::device::cuda::synchronize(0).unwrap();
            let t0 = std::time::Instant::now();
            for _ in 0..iters { run(); }
            synaptix_core::device::cuda::synchronize(0).unwrap();
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let fl = 2.0 * m as f64 * n as f64 * k as f64;
            println!("m={m:5} n={n:5} k={k:5} BF16 dense bias={}: {:7.2}ms = {:6.1} TF",
                if with_bias { "y" } else { "n" }, dt * 1e3, fl / dt / 1e12);
        }
    }
}

//! Во что упирается MoE на префилле: NVFP4-GEMM на формах одного эксперта.
//!
//! На промпте в 6k токенов каждый эксперт слоя получает около сотни строк, и
//! таких умножений выходит два десятка тысяч на чанк. Здесь замеряется, сколько
//! из них можно сделать в секунду и какая доля пиковой пропускной способности
//! при этом достигается.

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) - 0.5
        })
        .collect()
}

fn bench(m: usize, n: usize, k: usize, iters: usize) {
    let device = Device::Cuda(0);
    let w = Tensor::from_vec::<_, f32>(noise(1, n * k), vec![n, k], Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F16))
        .and_then(|t| t.to_device(device))
        .expect("вес");
    let qw = w.quantize_to_nvfp4().expect("квант веса");
    let x = Tensor::from_vec::<_, f32>(noise(2, m * k), vec![m, k], Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F16))
        .and_then(|t| t.to_device(device))
        .expect("активации");

    for _ in 0..3 {
        let _ = x.linear_quant(&qw).expect("прогрев");
    }
    synaptix_core::device::cuda::default_stream(0)
        .and_then(|s| s.synchronize().map_err(|e| synaptix_core::error::SynaptixError::Cuda(format!("{e:?}"))))
        .expect("sync");

    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = x.linear_quant(&qw).expect("gemm");
    }
    synaptix_core::device::cuda::default_stream(0)
        .and_then(|s| s.synchronize().map_err(|e| synaptix_core::error::SynaptixError::Cuda(format!("{e:?}"))))
        .expect("sync");
    let dt = t0.elapsed().as_secs_f64() / iters as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let bytes = (n * k / 2) as f64;
    println!(
        "M={m:4} N={n:5} K={k:5}: {:8.3} мс, {:6.1} TFLOPS, вес {:5.1} ГБ/с",
        dt * 1e3,
        flops / dt / 1e12,
        bytes / dt / 1e9
    );
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        println!("CUDA-устройств нет");
        return;
    }
    println!("gate_up эксперта Qwen4Exp (N=1280, K=2560)");
    for m in [1, 16, 64, 116, 256, 1024] {
        bench(m, 1280, 2560, 50);
    }
    println!("down эксперта (N=2560, K=640)");
    for m in [1, 16, 64, 116, 256, 1024] {
        bench(m, 2560, 640, 50);
    }
}

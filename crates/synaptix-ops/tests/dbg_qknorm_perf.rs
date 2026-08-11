//! Диагностика qk_norm ×15 (FLUX квант-путь): тайминг rms_norm на форме
//! [4608,24,128] F16 vs BF16 + прямой rms_norm_fused (ошибка = причина фолбэка).

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::rms_norm::rms_norm;

fn setup() -> bool {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).is_ok()
}

#[test]
fn qknorm_perf_f16_vs_bf16() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    for dt in [DType::BF16, DType::F16] {
        let x = Tensor::randn(vec![4608usize, 24, 128], Device::Cpu)
            .unwrap()
            .to_device(dev)
            .unwrap()
            .to_dtype(dt)
            .unwrap();
        let w = Tensor::ones(vec![128usize], dt, dev).unwrap();
        match x.rms_norm_fused(&w, 1e-6, false) {
            Ok(_) => println!("{dt:?}: rms_norm_fused OK"),
            Err(e) => println!("{dt:?}: rms_norm_fused ERR = {e}"),
        }
        for _ in 0..5 {
            let _ = rms_norm(&x, &w, 1e-6).unwrap();
        }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..50 {
            let y = rms_norm(&x, &w, 1e-6).unwrap();
            std::hint::black_box(&y);
        }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        println!("{dt:?}: rms_norm [4608,24,128] = {:.1}µs/вызов", t.elapsed().as_secs_f64() / 50.0 * 1e6);
    }
}

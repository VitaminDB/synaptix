//! Диагностика LTX v_attn1: flash_attention на формах stage1 (T=3520) и
//! stage2 (T=14080), [1,32,T,128] BF16. Эффективные TF = 4·T²·h·dh / t.
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
fn ltx_attn_shapes() {
    if !setup() {
        return;
    }
    let dev = Device::Cuda(0);
    for t in [3520usize, 14080] {
        let q = Tensor::randn(vec![1usize, 32, t, 128], Device::Cpu)
            .unwrap().to_device(dev).unwrap().to_dtype(DType::BF16).unwrap();
        let k = q.clone();
        let v = q.clone();
        let scale = 1.0f32 / (128f32).sqrt();
        let r = q.flash_attention(&k, &v, scale, false);
        match &r {
            Ok(_) => {}
            Err(e) => {
                println!("T={t}: flash ERR = {e}");
                continue;
            }
        }
        for _ in 0..3 {
            let _ = q.flash_attention(&k, &v, scale, false).unwrap();
        }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let n = if t > 8000 { 5 } else { 20 };
        let t0 = std::time::Instant::now();
        for _ in 0..n {
            let y = q.flash_attention(&k, &v, scale, false).unwrap();
            std::hint::black_box(&y);
        }
        synaptix_core::device::cuda::synchronize(0).unwrap();
        let dt = t0.elapsed().as_secs_f64() / n as f64;
        let fl = 4.0 * (t as f64) * (t as f64) * 32.0 * 128.0;
        println!("T={t}: flash {:.2}ms = {:.1} TF eff", dt * 1e3, fl / dt / 1e12);
    }
}

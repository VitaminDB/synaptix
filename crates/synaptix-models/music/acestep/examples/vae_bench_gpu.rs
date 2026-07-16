//! Изолированный GPU VAE decode-bench (для профилирования без AR/DiT/offload).
//! Фикс. латент [1,64,T], decode_tiled N раз. Запуск:
//!   cargo run --profile fast-release -p synaptix-music-acestep --example vae_bench_gpu
//! Длина латента: VAE_BENCH_T (дефолт 750 ≈ 30с), число прогонов: VAE_BENCH_N.

use std::path::Path;
use std::time::Instant;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::vae::AceStepVae;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let vae = AceStepVae::open(
        Path::new("storage/syn_models/acestep_vae.syn"),
        device,
    )
    .expect("open vae");

    let t: usize = std::env::var("VAE_BENCH_T").ok().and_then(|s| s.parse().ok()).unwrap_or(750);
    let c = 64usize;
    // детерминированный псевдо-латент [1, 64, T]
    let mut d = 0x1234_5678u64;
    let data: Vec<f32> = (0..c * t)
        .map(|_| {
            d = d.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((d >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect();
    let lat = Tensor::from_vec(data, vec![1usize, c, t], device).unwrap();

    let n: usize = std::env::var("VAE_BENCH_N").ok().and_then(|s| s.parse().ok()).unwrap_or(4);
    for i in 0..n {
        let t0 = Instant::now();
        let out = vae.decode_tiled(&lat, 500, 32).expect("decode");
        // sync через readback одного значения
        let _ = out.narrow(2, 0, 1).unwrap().contiguous().unwrap().to_dtype(DType::F32).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        eprintln!("[vae-bench] run {i}: {:.3}s out={:?}", t0.elapsed().as_secs_f32(), out.dims());
    }
}

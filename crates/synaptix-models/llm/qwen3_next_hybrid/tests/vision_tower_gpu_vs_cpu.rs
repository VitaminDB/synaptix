//! Гейтед: башня зрения Qwen3.8 на карте (F16/BF16) против CPU F32 на одной
//! картинке — косинус строк эмбеддингов.
//! `SYN_QWEN38_BUNDLE=… SYN_QWEN38_IMAGE=… cargo test -p synaptix-llm-qwen3-next-hybrid --release --test vision_tower_gpu_vs_cpu -- --nocapture`

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_vlm_qwen3::{load_from_bundle, prepare_image, PreprocessLimits};

fn rows(path: &PathBuf, img: &PathBuf, dev: Device, dt: DType) -> Vec<Vec<f32>> {
    let tower = load_from_bundle(path, dev, dt).expect("load");
    let mut lim = PreprocessLimits::default();
    let f = tower.config.size_factor();
    lim.max_pixels = lim.max_pixels.min(256 * f * f);
    lim.min_pixels = lim.min_pixels.min(lim.max_pixels);
    let p = prepare_image(img, &tower.config, lim, dev).expect("prep");
    let e = no_grad(|| tower.forward(&p.patches, p.grid)).expect("fwd");
    e.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().to_vec2::<f32>().unwrap()
}

#[test]
fn tower_gpu_matches_cpu() {
    let (Some(b), Some(i)) = (std::env::var_os("SYN_QWEN38_BUNDLE"), std::env::var_os("SYN_QWEN38_IMAGE")) else {
        eprintln!("пропуск");
        return;
    };
    let (b, i) = (PathBuf::from(b), PathBuf::from(i));
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let cpu = rows(&b, &i, Device::Cpu, DType::F32);
    for dt in [DType::F16, DType::BF16] {
        let gpu = rows(&b, &i, Device::Cuda(0), dt);
        assert_eq!(cpu.len(), gpu.len());
        let mut worst = 1f32;
        let mut sum = 0f32;
        for (a, g) in cpu.iter().zip(&gpu) {
            let d: f32 = a.iter().zip(g).map(|(x, y)| x * y).sum();
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let ng: f32 = g.iter().map(|x| x * x).sum::<f32>().sqrt();
            let c = d / (na * ng + 1e-12);
            worst = worst.min(c);
            sum += c;
        }
        eprintln!("{dt:?}: rows={} cos mean={:.4} min={:.4}", cpu.len(), sum / cpu.len() as f32, worst);
    }
}

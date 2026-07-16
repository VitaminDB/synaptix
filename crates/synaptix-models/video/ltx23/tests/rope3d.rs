//! Изоляция 3D SPLIT-RoPE: сравнение `rope3d` с Python `precompute_freqs_cis`.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;

const REFDIR: &str = "tests/reference_data/ltx_gemma";

#[test]
fn rope3d_matches_python() {
    let pos_path = format!("{REFDIR}/dit_video_ref.safetensors");
    let rope_path = format!("{REFDIR}/rope3d_ref.safetensors");
    if !Path::new(&pos_path).exists() || !Path::new(&rope_path).exists() {
        eprintln!("skip rope3d_matches_python: refs absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;
    let positions: Vec<f64> = SafetensorsLoader::open(&pos_path).unwrap()
        .load("positions").unwrap() // [3,T,2] f32
        .reshape(vec![3 * 64 * 2]).unwrap()
        .to_vec1::<f32>().unwrap().iter().map(|&x| x as f64).collect();
    let rl = SafetensorsLoader::open(&rope_path).unwrap();
    let cos_ref: Vec<f32> = rl.load("cos").unwrap().reshape(vec![32 * 64 * 64]).unwrap().to_vec1::<f32>().unwrap();
    let sin_ref: Vec<f32> = rl.load("sin").unwrap().reshape(vec![32 * 64 * 64]).unwrap().to_vec1::<f32>().unwrap();

    let (cos, sin) = VideoDit::_rope3d_for_test(&positions, 64, 32, 128, 10000.0, &[20.0, 2048.0, 2048.0], dev, DType::F32).unwrap();
    let cos_o: Vec<f32> = cos.reshape(vec![32 * 64 * 64]).unwrap().to_vec1::<f32>().unwrap();
    let sin_o: Vec<f32> = sin.reshape(vec![32 * 64 * 64]).unwrap().to_vec1::<f32>().unwrap();

    let mut max_cos = 0.0f32;
    let mut max_sin = 0.0f32;
    for i in 0..cos_ref.len() {
        max_cos = max_cos.max((cos_o[i] - cos_ref[i]).abs());
        max_sin = max_sin.max((sin_o[i] - sin_ref[i]).abs());
    }
    eprintln!("rope3d max|Δcos|={max_cos:.2e} max|Δsin|={max_sin:.2e}");
    assert!(max_cos < 1e-4 && max_sin < 1e-4, "rope3d mismatch: cos {max_cos}, sin {max_sin}");
}

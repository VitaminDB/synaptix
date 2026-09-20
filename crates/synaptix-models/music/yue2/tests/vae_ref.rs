//! Сверка Oobleck-декодера с эталоном релиза.
//!
//! Латент строится формулой (без RNG), поэтому тот же вход легко повторить в
//! Python: `scratchpad/ref/vae_ref.py`. Числа ниже — из эталонного прогона
//! (FP32, CPU) на бандле `yue2-vae.syn`.

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_yue2::vae::Yue2Vae;

const FRAMES: usize = 32;
const LATENT: usize = 64;

/// Эталон: `samples = 1920·кадры − 64`.
const REF_SAMPLES: usize = 61376;
const REF_RMS: f32 = 0.179_861_0;
const REF_MIN: f32 = -0.864_993_3;
const REF_MAX: f32 = 0.798_544_8;
const REF_FIRST8_L: [f32; 8] = [
    -0.020_512_132,
    -0.036_233_991,
    -0.013_507_753,
    -0.008_221_379,
    0.013_715_261,
    0.010_511_511,
    0.014_921_346,
    0.009_292_081,
];
const REF_FIRST8_R: [f32; 8] = [
    -0.016_953_267,
    -0.021_810_796,
    -0.002_909_073,
    0.005_885_845,
    0.023_821_404,
    0.024_310_224,
    0.030_606_631,
    0.030_603_791,
];

fn vae_path() -> Option<PathBuf> {
    let p = std::env::var("SYN_YUE2_VAE").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join("Storage/syn_models/yue2-vae.syn")
    });
    p.exists().then_some(p)
}

/// Тот же латент, что у эталонного скрипта.
fn det_latent() -> Vec<f32> {
    let mut z = vec![0f32; LATENT * FRAMES];
    for c in 0..LATENT {
        for t in 0..FRAMES {
            z[c * FRAMES + t] =
                (0.1 * (c as f32 + 1.0) + 0.37 * t as f32).sin() * (1.0 + 0.01 * c as f32);
        }
    }
    z
}

#[test]
fn decode_matches_reference() {
    let Some(path) = vae_path() else {
        eprintln!("[yue2-vae] бандла нет — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let vae = Yue2Vae::open(&path, Device::Cpu, DType::F32, true).expect("открыть VAE");
    assert_eq!(vae.output_len(FRAMES), REF_SAMPLES, "длина полного декода");

    let z = Tensor::from_vec(det_latent(), vec![1usize, LATENT, FRAMES], Device::Cpu)
        .expect("латент");
    let audio = vae.decode(&z).expect("декод");
    assert_eq!(audio.dims(), &[1, 2, REF_SAMPLES]);

    let flat: Vec<f32> = audio.flatten_all().unwrap().to_vec1().unwrap();
    assert!(flat.iter().all(|v| v.is_finite()), "в аудио есть не-конечные значения");
    let left = &flat[..REF_SAMPLES];
    let right = &flat[REF_SAMPLES..];

    let rms = (flat.iter().map(|v| v * v).sum::<f32>() / flat.len() as f32).sqrt();
    let min = flat.iter().copied().fold(f32::INFINITY, f32::min);
    let max = flat.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    eprintln!("[yue2-vae] rms={rms:.6} min={min:.6} max={max:.6}");
    assert!((rms - REF_RMS).abs() < 2e-3, "rms {rms} против эталона {REF_RMS}");
    assert!((min - REF_MIN).abs() < 5e-3, "min {min} против эталона {REF_MIN}");
    assert!((max - REF_MAX).abs() < 5e-3, "max {max} против эталона {REF_MAX}");

    for (i, (&got, &want)) in left.iter().zip(REF_FIRST8_L.iter()).enumerate() {
        assert!((got - want).abs() < 2e-3, "L[{i}]: {got} против эталона {want}");
    }
    for (i, (&got, &want)) in right.iter().zip(REF_FIRST8_R.iter()).enumerate() {
        assert!((got - want).abs() < 2e-3, "R[{i}]: {got} против эталона {want}");
    }
}

#[test]
fn tiled_decode_equals_full() {
    let Some(path) = vae_path() else {
        eprintln!("[yue2-vae] бандла нет — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let vae = Yue2Vae::open(&path, Device::Cpu, DType::F32, true).expect("открыть VAE");
    let z = Tensor::from_vec(det_latent(), vec![1usize, LATENT, FRAMES], Device::Cpu)
        .expect("латент");
    let full: Vec<f32> = vae.decode(&z).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let tiled = vae
        .decode_tiled(&z, 16, 16, &|| false, &|_, _| {})
        .expect("тайлами");
    assert_eq!(tiled.dims(), &[1, 2, REF_SAMPLES], "тайлинг меняет длину");
    let tiled: Vec<f32> = tiled.flatten_all().unwrap().to_vec1().unwrap();
    let diff = full
        .iter()
        .zip(tiled.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("[yue2-vae] тайлы против полного декода: max|Δ| = {diff:e}");
    // Ядра тайлов считаются с полным контекстом — расхождение только от
    // порядка операций во свёртке, не от склейки.
    assert!(diff < 1e-5, "тайлинг разошёлся с полным декодом: {diff}");
}

//! GPU: `encode_mean_tiled` на длинном аудио. Целиком `encode_mean` на треке
//! в несколько минут падал на запуске GEMM (M = длина в сэмплах вылезала за
//! grid.y) и не влез бы в VRAM — окна должны давать тот же латент, что и
//! целиком, там, где целиком ещё можно.

use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::vae::{AceStepVae, ENCODE_CANCELLED};

fn vae_path() -> Option<PathBuf> {
    [
        PathBuf::from("storage/syn_models/acestep_vae.syn"),
        PathBuf::from("/home/master/Storage/syn_models/acestep_vae.syn"),
    ]
    .into_iter()
    .find(|p| p.exists())
}

fn open_gpu() -> Option<AceStepVae> {
    let path = vae_path()?;
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    synaptix_core::device::cuda::get(0).ok()?;
    Some(AceStepVae::open(&path, Device::Cuda(0)).expect("open vae"))
}

/// Стерео-«музыка» [1, 2, secs·48k]: пара синусов с биением и шум, чтобы у
/// латента было что кодировать на всей длине.
fn stereo(secs: f32) -> Tensor {
    let n = (secs * 48_000.0) as usize;
    let mut s = 0x2545_F491_4F6C_DD1Du64;
    let mut noise = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05
    };
    let mut data = Vec::with_capacity(2 * n);
    for ch in 0..2 {
        for i in 0..n {
            let t = i as f32 / 48_000.0;
            let f = if ch == 0 { 220.0 } else { 330.0 };
            let env = 0.5 + 0.5 * (t * 0.7).sin();
            data.push(0.3 * env * (std::f32::consts::TAU * f * t).sin() + noise());
        }
    }
    Tensor::from_vec(data, vec![1usize, 2, n], Device::Cuda(0)).unwrap()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap()
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += *x as f64 * *x as f64;
        nb += *y as f64 * *y as f64;
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

#[test]
fn tiled_encode_matches_whole_encode() {
    let Some(vae) = open_gpu() else { return };
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let audio = stereo(24.0);
    let whole = vae.encode_mean(&audio).expect("whole encode");
    // Окна по 4 с ядра + 0,8 с контекста: 6 окон вместо одного.
    let tiled = vae.encode_mean_tiled(&audio, 100, 20, &|| false).expect("tiled encode");
    assert_eq!(tiled.dims(), whole.dims());
    let cos = cos_sim(&host(&tiled), &host(&whole));
    eprintln!("[vae tiled vs whole 24s] dims={:?} cos={cos:.6}", whole.dims());
    assert!(cos >= 0.999, "tiled encode drifts from whole encode: cos={cos}");
}

/// Сценарий чата «Vocal»: трек 3:43 целиком через ноду VAE Encode.
#[test]
fn tiled_encode_handles_multi_minute_track() {
    let Some(vae) = open_gpu() else { return };
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let secs = 223.0;
    let audio = stereo(secs);
    let lat = vae.encode_mean_tiled(&audio, 750, 20, &|| false).expect("3:43 encode");
    let frames = ((secs * 48_000.0) as usize).div_ceil(1920);
    assert_eq!(lat.dims(), &[1, vae.config().decoder_input_channels, frames]);
    assert!(host(&lat).iter().all(|v| v.is_finite()), "latent has non-finite values");
}

#[test]
fn tiled_encode_stops_on_cancel() {
    let Some(vae) = open_gpu() else { return };
    let _nograd = synaptix_core::grad::NoGradGuard::new();
    let audio = stereo(20.0);
    let calls = std::cell::Cell::new(0usize);
    let err = vae
        .encode_mean_tiled(&audio, 100, 20, &|| {
            calls.set(calls.get() + 1);
            calls.get() > 2
        })
        .expect_err("cancel must stop the encode");
    assert!(err.to_string().contains(ENCODE_CANCELLED), "unexpected error: {err}");
    assert_eq!(calls.get(), 3, "encode must stop at the first cancelled window");
}

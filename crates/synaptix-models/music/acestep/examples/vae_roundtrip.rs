//! Изоляция VAE: синус 440 Гц → encode_mean → decode → wav.
//! Если выход спектрально ≈ вход (доминанта 440) — VAE корректен; если шум — баг.
//! Запуск: cargo run --release -p synaptix-music-acestep --example vae_roundtrip

use std::path::Path;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::vae::AceStepVae;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let vae = AceStepVae::open(Path::new("storage/syn_models/acestep_vae.syn"), device)
        .expect("open vae");

    let sr = 48000usize;
    let secs = 2usize;
    let n = sr * secs;
    let freq = 440.0_f32;
    // stereo sine [1,2,N] (channels-first)
    let mut data = Vec::with_capacity(2 * n);
    for _ch in 0..2 {
        for i in 0..n {
            data.push((2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin() * 0.5);
        }
    }
    let audio = Tensor::from_vec(data, vec![1usize, 2, n], device).unwrap();

    let lat = vae.encode_mean(&audio).expect("encode");
    eprintln!("[vae-rt] latent dims {:?}", lat.dims());
    let out = vae.decode(&lat).expect("decode");
    eprintln!("[vae-rt] decoded dims {:?}", out.dims());

    let mono: Vec<f32> = out
        .narrow(1, 0, 1)
        .unwrap()
        .contiguous()
        .unwrap()
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    synaptix_audio::write_wav_mono_f32("/tmp/vae_roundtrip.wav", &mono, 48000).unwrap();
    let rms = (mono.iter().map(|x| x * x).sum::<f32>() / mono.len() as f32).sqrt();
    eprintln!("[vae-rt] wrote /tmp/vae_roundtrip.wav n={} rms={:.4}", mono.len(), rms);
}

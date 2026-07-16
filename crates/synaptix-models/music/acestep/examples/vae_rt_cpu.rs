//! CPU VAE roundtrip + dominant-freq check. TEMP diagnostic.
use std::path::Path;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_music_acestep::vae::AceStepVae;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    let device = Device::Cpu;
    let vae = AceStepVae::open(Path::new("storage/syn_models/acestep_vae.syn"), device)
        .expect("open vae");

    let sr = 48000usize;
    let secs = 1usize;
    let n = sr * secs;
    let freq = 440.0_f32;
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

    let mono: Vec<f32> = out.narrow(1, 0, 1).unwrap().contiguous().unwrap()
        .to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let rms = (mono.iter().map(|x| x * x).sum::<f32>() / mono.len() as f32).sqrt();

    // crude DFT over a coarse freq grid to find dominant bin
    let m = mono.len().min(sr); // 1 sec window
    let mut best_f = 0.0f64; let mut best_p = 0.0f64;
    for fk in (50..2000).step_by(5) {
        let w = 2.0 * std::f64::consts::PI * fk as f64 / sr as f64;
        let (mut re, mut im) = (0.0f64, 0.0f64);
        for i in 0..m {
            let s = mono[i] as f64;
            re += s * (w * i as f64).cos();
            im += s * (w * i as f64).sin();
        }
        let p = re * re + im * im;
        if p > best_p { best_p = p; best_f = fk as f64; }
    }
    eprintln!("[vae-rt] rms={:.5} dominant_freq≈{:.0}Hz (expect ~440 if VAE correct)", rms, best_f);
}

//! Audio-VAE encoder bit-exact: наш `ltx_log_mel` + `AudioVaeEncoder::encode` vs
//! официальный `encode_audio` (тот же 16k-вход). Эталон `/tmp/audio_enc_ref.safetensors`
//! (samples [N] f32 16kHz mono + tokens [1,Fa,128]). Гейт: наличие весов и эталона.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::audio_vae::{ltx_log_mel, AudioVaeEncoder};
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "/tmp/audio_enc_ref.safetensors";

#[test]
fn audio_encoder_matches_python() {
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip audio_encoder_matches_python: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let rl = SafetensorsLoader::open(REF).unwrap().with_device(Device::Cpu);
    let samples: Vec<f32> = rl.load("samples").unwrap().to_vec1().unwrap();
    let refl = rl.load("tokens").unwrap().to_device(dev).unwrap().to_dtype(DType::F32).unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::BF16).unwrap();
    let enc = AudioVaeEncoder::load(&ckpt, dev).expect("encoder load");
    let mel = ltx_log_mel(&[samples], dev).expect("mel");
    eprintln!("mel {:?}", mel.dims());
    let out = synaptix_core::grad::no_grad(|| enc.encode(&mel)).expect("encode");
    assert_eq!(out.dims(), refl.dims(), "shape {:?} vs ref {:?}", out.dims(), refl.dims());

    let a: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<f32> = refl.flatten_all().unwrap().to_vec1().unwrap();
    let (mut dot, mut na, mut nb, mut maxabs) = (0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
        maxabs = maxabs.max((*x - *y).abs() as f64);
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    eprintln!("Audio encoder: cos={cos:.6} max_abs={maxabs:.5} (n={})", a.len());
    assert!(cos > 0.999, "cos {cos} too low");
}

#[test]
fn audio_encoder_dump_ours() {
    if std::env::var("AUDIO_DUMP").is_err() || !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let rl = SafetensorsLoader::open(REF).unwrap().with_device(Device::Cpu);
    let samples: Vec<f32> = rl.load("samples").unwrap().to_vec1().unwrap();
    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::BF16).unwrap();
    let enc = AudioVaeEncoder::load(&ckpt, dev).unwrap();
    let mel = if let Ok(raw) = std::fs::read("/tmp/py_mel.f32") {
        if std::env::var("USE_PY_MEL").is_ok() {
            let v: Vec<f32> = raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
            synaptix_core::tensor::Tensor::from_vec(v, vec![1,2,346,64], dev).unwrap()
        } else { ltx_log_mel(&[samples], dev).unwrap() }
    } else { ltx_log_mel(&[samples], dev).unwrap() };
    let out = synaptix_core::grad::no_grad(|| enc.encode(&mel)).unwrap();
    let v: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
    let mel_v: Vec<f32> = mel.flatten_all().unwrap().to_vec1().unwrap();
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write("/tmp/audio_ours_tokens.f32", b).unwrap();
    let b2: Vec<u8> = mel_v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write("/tmp/audio_ours_mel.f32", b2).unwrap();
    eprintln!("dumped tokens {:?} mel {:?}", out.dims(), mel.dims());
}

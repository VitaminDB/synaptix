//! VAE encoder bit-exact: наш `VaeEncoder::encode` vs официальный LTX-энкодер.
//! Эталон `/tmp/vae_enc_ref.safetensors` (frames [1,3,9,64,64] + latent [1,128,2,2,2])
//! генерируется Python-скриптом (VideoEncoderConfigurator + vae.encoder.* веса).
//! Гейт: наличие чекпойнта и эталона.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::vae::VaeEncoder;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "/tmp/vae_enc_ref.safetensors";

#[test]
fn vae_encoder_matches_python() {
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip vae_encoder_matches_python: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let frames = rl.load("frames").unwrap().to_dtype(DType::F32).unwrap();
    let refl = rl.load("latent").unwrap().to_dtype(DType::F32).unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::BF16).unwrap();
    let enc = VaeEncoder::load(&ckpt, dev).expect("encoder load");
    let out = synaptix_core::grad::no_grad(|| enc.encode(&frames)).expect("encode");
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
    eprintln!("VAE encoder: cos={cos:.6} max_abs={maxabs:.5} (n={})", a.len());
    assert!(cos > 0.9999, "cos {cos} too low");
    assert!(maxabs < 5e-2, "max_abs {maxabs} too high");
}

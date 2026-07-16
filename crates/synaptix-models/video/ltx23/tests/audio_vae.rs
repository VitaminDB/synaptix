//! Фаза 9: Audio VAE decode против эталона LTX AudioDecoder. Латент → log-mel.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::audio_vae::AudioVaeDecoder;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/audio_vae_ref.safetensors";

#[test]
fn audio_vae_decode_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip audio_vae_decode_matches: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = if std::env::var("SYN_LTX_VAE_CPU").is_ok() {
        Device::Cpu
    } else {
        synaptix_kernels_cuda::ensure_registered();
        Device::Cuda(0)
    };

    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let latent = rl.load("latent").unwrap(); // [8,4,16]
    let (c, f, m) = (latent.dims()[0], latent.dims()[1], latent.dims()[2]);
    let latent = latent.reshape(vec![1, c, f, m]).unwrap();
    let mel_ref_t = rl.load("mel").unwrap();
    let rd = mel_ref_t.dims().to_vec();
    let n: usize = rd.iter().product();
    let mel_ref: Vec<f32> = mel_ref_t.reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::F32).unwrap();
    let dec = AudioVaeDecoder::load(&ckpt, dev).expect("load audio vae");
    let mel = synaptix_core::grad::no_grad(|| dec.decode(&latent)).expect("decode");
    assert_eq!(mel.dims(), &[1, rd[0], rd[1], rd[2]], "mel shape");
    let ours: Vec<f32> = mel.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    let (mut dot, mut nr, mut no, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (r, o) = (mel_ref[i] as f64, ours[i] as f64);
        dot += r * o; nr += r * r; no += o * o; mx = mx.max((r - o).abs());
    }
    let cos = dot / (nr.sqrt() * no.sqrt() + 1e-12);
    eprintln!("audio vae decode: cos={cos:.6} max|Δ|={mx:.4e} (mel {:?})", rd);
    assert!(cos > 0.999, "audio vae cos: {cos}");
    assert!(mx < 5e-2, "audio vae max|Δ|: {mx}");
}

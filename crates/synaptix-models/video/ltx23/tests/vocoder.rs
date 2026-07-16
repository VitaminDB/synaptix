//! Валидация base BigVGAN-v2 вокодера (16кГц) против LTX `Vocoder`. f32.
//! Reference: vocoder_ref.safetensors (mel [1,2,40,64] → base_16k [1,2,6400]).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::vocoder::{BaseVocoder, VocoderWithBwe};

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/vocoder_ref.safetensors";

fn flat(t: &Tensor) -> Vec<f32> {
    let n: usize = t.dims().iter().product();
    t.contiguous().unwrap().reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap()
}

fn cos_maxabs(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut dot, mut na, mut nb, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        mx = mx.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx)
}

#[test]
fn base_vocoder_matches_ltx() {
    if !Path::new(REF).exists() || !Path::new(CKPT).exists() {
        eprintln!("skip base_vocoder: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;

    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let mel = rl.load("mel").unwrap(); // [1,2,40,64]
    let want = flat(&rl.load("base_16k").unwrap());

    let voc = BaseVocoder::load(CKPT, dev).expect("base vocoder load");
    let got = synaptix_core::grad::no_grad(|| voc.forward(&mel)).expect("forward");
    eprintln!("base out {:?}", got.dims());

    let (cos, mx) = cos_maxabs(&flat(&got), &want);
    eprintln!("base vocoder: cos={cos:.6} max|Δ|={mx:.3e} (n={})", want.len());
    assert!(cos > 0.999, "base vocoder cos={cos}");
}

#[test]
fn full_vocoder_bwe_matches_ltx() {
    if !Path::new(REF).exists() || !Path::new(CKPT).exists() {
        eprintln!("skip full_vocoder: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;

    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let mel = rl.load("mel").unwrap(); // [1,2,40,64]
    let want = flat(&rl.load("full_48k").unwrap());

    let voc = VocoderWithBwe::load(CKPT, dev).expect("bwe vocoder load");
    let got = synaptix_core::grad::no_grad(|| voc.forward(&mel)).expect("forward");
    eprintln!("full out {:?}", got.dims());

    let (cos, mx) = cos_maxabs(&flat(&got), &want);
    eprintln!("full vocoder+bwe: cos={cos:.6} max|Δ|={mx:.3e} (n={})", want.len());
    assert!(cos > 0.999, "full vocoder cos={cos}");
}

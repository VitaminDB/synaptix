//! Фаза 5: Video VAE decode против эталона LTX VideoDecoder. Латент → RGB.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::vae::VaeDecoder;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/vae_decode_ref.safetensors";

#[test]
fn vae_decode_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip vae_decode_matches: weights/ref absent");
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
    let latent = rl.load("latent").unwrap(); // [128,2,8,8] f32
    let (c, f, h, w) = (latent.dims()[0], latent.dims()[1], latent.dims()[2], latent.dims()[3]);
    let latent = latent.reshape(vec![1, c, f, h, w]).unwrap();
    let rgb_ref_t = rl.load("rgb").unwrap(); // [3,9,256,256]
    let rd = rgb_ref_t.dims().to_vec();
    let n: usize = rd.iter().product();
    let rgb_ref: Vec<f32> = rgb_ref_t.reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::F32).unwrap();
    let dec = VaeDecoder::load(&ckpt, dev).expect("load vae");
    let rgb = synaptix_core::grad::no_grad(|| dec.decode(&latent)).expect("decode");
    assert_eq!(rgb.dims(), &[1, rd[0], rd[1], rd[2], rd[3]], "rgb shape");
    let ours: Vec<f32> = rgb.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    let (mut dot, mut nr, mut no, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (r, o) = (rgb_ref[i] as f64, ours[i] as f64);
        dot += r * o; nr += r * r; no += o * o; mx = mx.max((r - o).abs());
    }
    let cos = dot / (nr.sqrt() * no.sqrt() + 1e-12);
    eprintln!("vae decode: cos={cos:.6} max|Δ|={mx:.4e} (rgb {:?})", rd);
    assert!(cos > 0.999, "vae decode cos: {cos}");
    assert!(mx < 5e-2, "vae decode max|Δ|: {mx}");
}

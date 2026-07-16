//! Фаза 7: spatial upscaler ×2 (LatentUpsampler) против эталона LTX upsample_video.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::upscaler::Upsampler;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const UP: &str = "models/ltx2.3_v1.1/ltx-2.3-spatial-upscaler-x2-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/upscaler_ref.safetensors";

#[test]
fn upscaler_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(UP).exists() || !Path::new(REF).exists() || !Path::new(CKPT).exists() {
        eprintln!("skip upscaler_matches: weights/ref absent");
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
    let latent = rl.load("latent").unwrap(); // [128,2,8,8]
    let (c, f, h, w) = (latent.dims()[0], latent.dims()[1], latent.dims()[2], latent.dims()[3]);
    let latent = latent.reshape(vec![1, c, f, h, w]).unwrap();
    let up_ref_t = rl.load("up").unwrap();
    let rd = up_ref_t.dims().to_vec();
    let n: usize = rd.iter().product();
    let up_ref: Vec<f32> = up_ref_t.reshape(vec![n]).unwrap().to_vec1::<f32>().unwrap();

    // VAE-статистики из основного чекпойнта
    let ml = SafetensorsLoader::open(CKPT).unwrap().with_device(dev);
    let mean = ml.load("vae.per_channel_statistics.mean-of-means").unwrap();
    let std = ml.load("vae.per_channel_statistics.std-of-means").unwrap();

    let up = Upsampler::load(UP, &mean, &std, dev).expect("load upscaler");
    let out = synaptix_core::grad::no_grad(|| up.upsample(&latent)).expect("upsample");
    assert_eq!(out.dims(), &[1, rd[0], rd[1], rd[2], rd[3]], "up shape");
    let ours: Vec<f32> = out.reshape(vec![n]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    let (mut dot, mut nr, mut no, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (r, o) = (up_ref[i] as f64, ours[i] as f64);
        dot += r * o; nr += r * r; no += o * o; mx = mx.max((r - o).abs());
    }
    let cos = dot / (nr.sqrt() * no.sqrt() + 1e-12);
    eprintln!("upscaler: cos={cos:.6} max|Δ|={mx:.4e} (up {:?})", rd);
    assert!(cos > 0.999, "upscaler cos: {cos}");
    assert!(mx < 5e-2, "upscaler max|Δ|: {mx}");
}

//! Пространственный тайлинговый VAE-декод == не-тайлинговый (бесшовность).
//! Гейт SYN_LTX_GEMMA. Halo из SYN_LTX_VAE_HALO (деф. 8).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::vae::VaeDecoder;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";

#[test]
fn vae_tiled_matches_untiled() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() {
        eprintln!("skip: weights absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::F32).unwrap();
    let dec = VaeDecoder::load(&ckpt, dev).expect("load vae");

    // детерминированный латент [1,128,4,24,24] (большой spatial для тайлинга 2×2)
    let (c, fp, hp, wp) = (128usize, 4usize, 24usize, 24usize);
    let n = c * fp * hp * wp;
    let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 0.7).collect();
    let latent = Tensor::from_vec(data, vec![1, c, fp, hp, wp], dev).unwrap();

    let decode = |nh: usize, nw: usize| -> Tensor {
        synaptix_video_ltx23::runtime::set_vae_grid(Some((nh, nw)));
        synaptix_core::grad::no_grad(|| dec.decode(&latent)).expect("decode")
    };
    let untiled = decode(1, 1);
    let tiled = decode(2, 2); // 2×2 spatial tiles + halo

    assert_eq!(untiled.dims(), tiled.dims(), "форма tiled != untiled");
    let (f, h, w) = (untiled.dims()[2], untiled.dims()[3], untiled.dims()[4]);
    let u: Vec<f32> = untiled.reshape(vec![3 * f * h * w]).unwrap().to_vec1::<f32>().unwrap();
    let t: Vec<f32> = tiled.reshape(vec![3 * f * h * w]).unwrap().to_vec1::<f32>().unwrap();

    let (mut dot, mut nu, mut nt, mut mx) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..u.len() {
        let (a, b) = (u[i] as f64, t[i] as f64);
        dot += a * b;
        nu += a * a;
        nt += b * b;
        mx = mx.max((a - b).abs());
    }
    let cos = dot / (nu.sqrt() * nt.sqrt() + 1e-12);
    eprintln!("spatial-tiled(2×2) vs untiled: out {h}×{w}, cos={cos:.6} max|Δ|={mx:.4e}");
    assert!(cos > 0.999, "бесшовность нарушена: cos={cos}");
}

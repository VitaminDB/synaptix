//! Изоляция ОДНОГО блока DiT + head (dense f32, CPU — быстро, без шума кванта)
//! против Python 1-блочного эталона. Локализует баг блока/head отдельно от
//! накопления по 48 блокам и от MXFP8.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const INP: &str = "tests/reference_data/ltx_gemma/dit_video_ref.safetensors";
const REF1: &str = "tests/reference_data/ltx_gemma/dit_video_1blk.safetensors";

#[test]
fn video_dit_1block_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF1).exists() {
        eprintln!("skip video_dit_1block_matches: weights/ref absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(Some(1));
    synaptix_kernels_cpu::ensure_registered();
    let dev = Device::Cpu;

    let il = SafetensorsLoader::open(INP).unwrap().with_device(dev);
    let latent = il.load("latent").unwrap();
    let t = latent.dims()[0];
    let latent = latent.reshape(vec![1, t, 128]).unwrap().to_dtype(DType::F32).unwrap();
    let timesteps: Vec<f32> = il.load("timesteps").unwrap().to_vec1::<f32>().unwrap();
    let sigma = il.load("sigma").unwrap().to_vec1::<f32>().unwrap()[0];
    let positions: Vec<f64> = il.load("positions").unwrap().reshape(vec![3 * t * 2]).unwrap()
        .to_vec1::<f32>().unwrap().iter().map(|&x| x as f64).collect();
    let ttxt = il.load("context").unwrap().dims()[0];
    let context = il.load("context").unwrap().reshape(vec![1, ttxt, 4096]).unwrap().to_dtype(DType::F32).unwrap();
    let ref_v: Vec<f32> = SafetensorsLoader::open(REF1).unwrap()
        .load("velocity").unwrap().reshape(vec![t * 128]).unwrap().to_vec1::<f32>().unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::F32).unwrap();
    let dit = VideoDit::load(&ckpt, dev, DType::F32, DType::F32).expect("load 1-block dit");
    let vx = synaptix_core::grad::no_grad(|| dit.forward(&latent, &timesteps, sigma, &positions, &context)).expect("forward");
    let ours: Vec<f32> = vx.reshape(vec![t * 128]).unwrap().to_vec1::<f32>().unwrap();

    let hd = 128usize;
    let mut min_cos = f32::INFINITY;
    let mut max_rel = 0.0f32;
    for p in 0..t {
        let (mut dot, mut nr, mut no, mut dl2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..hd {
            let r = ref_v[p * hd + k] as f64;
            let o = ours[p * hd + k] as f64;
            dot += r * o; nr += r * r; no += o * o; dl2 += (r - o) * (r - o);
        }
        let cos = (dot / (nr.sqrt() * no.sqrt() + 1e-12)) as f32;
        if cos < min_cos { min_cos = cos; }
        max_rel = max_rel.max((dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32);
    }
    eprintln!("1-block velocity: per-row min_cos={min_cos:.5} max_rel={max_rel:.4}");
    assert!(min_cos > 0.999, "1-block velocity cos: {min_cos}");
}

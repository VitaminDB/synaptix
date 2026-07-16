//! Проверка GPU dense bf16 (best_cu float-acc) на N блоках против bf16-эталона.
//! Если cos высокий — подтверждает offload-путь (GPU dense bf16 точен, в отличие
//! от CPU bf16-acc 0.21 и MXFP8 f16-acc 0.62). N из SYN_LTX_DIT_NBLOCKS (деф. 8).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const INP: &str = "tests/reference_data/ltx_gemma/dit_video_ref.safetensors";

#[test]
fn video_dit_nblock_gpu_dense() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    let n: usize = std::env::var("SYN_LTX_DIT_NBLOCKS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let ref_path = format!("tests/reference_data/ltx_gemma/dit_video_{n}blk_bf16.safetensors");
    if !Path::new(CKPT).exists() || !Path::new(&ref_path).exists() {
        eprintln!("skip: weights/ref absent ({ref_path})");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(Some(n));
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let il = SafetensorsLoader::open(INP).unwrap().with_device(dev);
    let latent = il.load("latent").unwrap();
    let t = latent.dims()[0];
    let latent = latent.reshape(vec![1, t, 128]).unwrap().to_dtype(DType::BF16).unwrap();
    let timesteps: Vec<f32> = il.load("timesteps").unwrap().to_vec1::<f32>().unwrap();
    let sigma = il.load("sigma").unwrap().to_vec1::<f32>().unwrap()[0];
    let positions: Vec<f64> = il.load("positions").unwrap().reshape(vec![3 * t * 2]).unwrap()
        .to_vec1::<f32>().unwrap().iter().map(|&x| x as f64).collect();
    let ttxt = il.load("context").unwrap().dims()[0];
    let context = il.load("context").unwrap().reshape(vec![1, ttxt, 4096]).unwrap().to_dtype(DType::BF16).unwrap();
    let ref_v: Vec<f32> = SafetensorsLoader::open(&ref_path).unwrap()
        .load("velocity").unwrap().reshape(vec![t * 128]).unwrap().to_vec1::<f32>().unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::BF16).unwrap();
    let dit = VideoDit::load(&ckpt, dev, DType::BF16, DType::BF16).expect("load"); // dense bf16
    let vx = synaptix_core::grad::no_grad(|| dit.forward(&latent, &timesteps, sigma, &positions, &context)).expect("forward");
    let ours: Vec<f32> = vx.reshape(vec![t * 128]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    let hd = 128usize;
    let mut min_cos = f32::INFINITY;
    let mut max_rel = 0.0f32;
    for p in 0..t {
        let (mut dot, mut nr, mut no, mut dl2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..hd {
            let (r, o) = (ref_v[p * hd + k] as f64, ours[p * hd + k] as f64);
            dot += r * o; nr += r * r; no += o * o; dl2 += (r - o) * (r - o);
        }
        min_cos = min_cos.min((dot / (nr.sqrt() * no.sqrt() + 1e-12)) as f32);
        max_rel = max_rel.max((dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32);
    }
    eprintln!("{n}-block GPU dense bf16 velocity: per-row min_cos={min_cos:.5} max_rel={max_rel:.4}");
    assert!(min_cos > 0.99, "{n}-block GPU dense bf16 cos: {min_cos}");
}

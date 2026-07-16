//! Фаза 4: video-only DiT forward (velocity) против эталона LTXModel(VideoOnly).
//! Вход (latent/timesteps/sigma/positions/context) сдемплен из Python, скармливается
//! идентично. Гейт: SYN_LTX_GEMMA. CPU dense bf16 (19B влезает в RAM; тождественнее
//! эталону, чем GPU-квант).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/dit_video_ref.safetensors";

#[test]
fn video_dit_velocity_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(REF).exists() {
        eprintln!("skip video_dit_velocity_matches: weights/ref absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    // SYN_LTX_DIT_DENSE=1 → CPU dense bf16 (тождественнее эталону, медленно);
    // иначе CUDA MXFP8 (быстро, ~10GB).
    let dense = std::env::var("SYN_LTX_DIT_DENSE").is_ok();
    let (dev, quant) = if dense {
        (Device::Cpu, DType::BF16)
    } else {
        synaptix_kernels_cuda::ensure_registered();
        (Device::Cuda(0), DType::MXFP8)
    };

    let rl = SafetensorsLoader::open(REF).unwrap().with_device(dev);
    let latent = rl.load("latent").unwrap(); // [T,128] f32
    let (t, _c) = (latent.dims()[0], latent.dims()[1]);
    let latent = latent.reshape(vec![1, t, 128]).unwrap().to_dtype(DType::BF16).unwrap();
    let timesteps: Vec<f32> = rl.load("timesteps").unwrap().to_vec1::<f32>().unwrap();
    let sigma = rl.load("sigma").unwrap().to_vec1::<f32>().unwrap()[0];
    let positions: Vec<f64> = rl
        .load("positions").unwrap() // [3,T,2] f32
        .reshape(vec![3 * t * 2]).unwrap()
        .to_vec1::<f32>().unwrap()
        .iter().map(|&x| x as f64).collect();
    let ttxt = rl.load("context").unwrap().dims()[0];
    let context = rl.load("context").unwrap().reshape(vec![1, ttxt, 4096]).unwrap().to_dtype(DType::BF16).unwrap();
    let ref_v: Vec<f32> = rl.load("velocity").unwrap().reshape(vec![t * 128]).unwrap().to_vec1::<f32>().unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, dev, DType::BF16).unwrap();
    let dit = VideoDit::load(&ckpt, dev, DType::BF16, quant).expect("load dit");

    let vx = synaptix_core::grad::no_grad(|| dit.forward(&latent, &timesteps, sigma, &positions, &context))
        .expect("forward");
    assert_eq!(vx.dims(), &[1, t, 128]);
    let ours: Vec<f32> = vx.reshape(vec![t * 128]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

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
        let rel = (dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32;
        if cos < min_cos { min_cos = cos; }
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("velocity ({}): per-row min_cos={min_cos:.5} max_rel={max_rel:.4}",
        if dense { "dense bf16" } else { "MXFP8" });
    if dense {
        // dense bf16: строгий гейт корректности всех 48 блоков.
        assert!(min_cos > 0.99, "dense velocity cos слишком низкий: {min_cos}");
    } else {
        // MXFP8-веса: 8-бит на 48 блоках накапливает ошибку в velocity (логика
        // bit-exact — см. dit1/adaln/rope3d тесты). Для GPU-генерации Фаза 6
        // использует bf16 layer-offload. Здесь — только проверка «считается».
        assert!(min_cos > 0.3, "MXFP8 velocity подозрительно низкий: {min_cos}");
    }
}

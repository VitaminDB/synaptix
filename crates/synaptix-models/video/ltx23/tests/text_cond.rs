//! Фаза 3: видео-путь текст-кондишена (FeatureExtractorV2 + Embeddings1DConnector)
//! против эталона из официального LTX-кода. Вход — 49 hidden states Gemma (дамп
//! Фазы 2), выход — video_encoding. Изолирует Фазу 3 от Gemma. Гейт: SYN_LTX_GEMMA.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::text_encoder::{AudioTextConditioner, VideoTextConditioner};

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const HS: &str = "tests/reference_data/ltx_gemma/gemma_ref_s128_bf16.safetensors";
const REF: &str = "tests/reference_data/ltx_gemma/textcond_video_s128.safetensors";
const AREF: &str = "tests/reference_data/ltx_gemma/textcond_audio_s128.safetensors";

#[test]
fn video_text_conditioning_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(HS).exists() || !Path::new(REF).exists() {
        eprintln!("skip video_text_conditioning_matches: weights/refs absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    // SYN_TEXTCOND_F32=1: f32-прогон (разделение bf16-шума и логики)
    let tdt = if std::env::var("SYN_TEXTCOND_F32").as_deref() == Ok("1") { DType::F32 } else { DType::BF16 };

    // 49 hidden states [49,T,3840] + mask
    let hl = SafetensorsLoader::open(HS).unwrap().with_device(dev);
    let hs = hl.load("hidden_states").unwrap(); // [49,T,3840] f32 на dev
    let (n, t) = (hs.dims()[0], hs.dims()[1]);
    let mask_u32: Vec<u32> = SafetensorsLoader::open(HS)
        .unwrap()
        .load("attention_mask")
        .unwrap()
        .to_vec1::<i64>()
        .unwrap()
        .iter()
        .map(|&x| x as u32)
        .collect();
    let states: Vec<Tensor> = (0..n)
        .map(|i| {
            hs.narrow(0, i, 1).unwrap()
                .contiguous().unwrap() // [1,t,d]
                .to_dtype(DType::BF16).unwrap()
        })
        .collect();

    // эталон video_encoding [T,4096] (right-pad-reordered)
    let ref_v: Vec<f32> = SafetensorsLoader::open(REF)
        .unwrap()
        .load("video_encoding")
        .unwrap()
        .reshape(vec![t * 4096])
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    let cond = VideoTextConditioner::load(
        &LtxCheckpoint::open(CKPT, dev, tdt).unwrap(),
        dev,
        tdt,
    )
    .expect("load conditioner");

    let venc = synaptix_core::grad::no_grad(|| cond.forward(&states, &mask_u32)).expect("forward");
    assert_eq!(venc.dims(), &[1, t, 4096]);
    let ours: Vec<f32> = venc.reshape(vec![t * 4096]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    // per-row cos + rel-L2 по ВСЕМ T позициям (после register-substitution все
    // валидны: текст-токены + register-эмбеддинги).
    let h = 4096usize;
    let mut min_cos = f32::INFINITY;
    let mut max_rel = 0.0f32;
    let mut worst = 0usize;
    for p in 0..t {
        let (mut dot, mut nr, mut no, mut dl2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..h {
            let r = ref_v[p * h + k] as f64;
            let o = ours[p * h + k] as f64;
            dot += r * o; nr += r * r; no += o * o; dl2 += (r - o) * (r - o);
        }
        let cos = (dot / (nr.sqrt() * no.sqrt() + 1e-12)) as f32;
        let rel = (dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32;
        if cos < min_cos { min_cos = cos; worst = p; }
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("video_encoding: per-row min_cos={min_cos:.5} (worst row {worst}) max_rel={max_rel:.4}");
    assert!(min_cos > 0.99, "per-row cos слишком низкий: {min_cos}");
}

#[test]
fn audio_text_conditioning_matches() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(HS).exists() || !Path::new(AREF).exists() {
        eprintln!("skip audio_text_conditioning_matches: weights/refs absent");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let tdt = if std::env::var("SYN_TEXTCOND_F32").as_deref() == Ok("1") { DType::F32 } else { DType::BF16 };

    let hl = SafetensorsLoader::open(HS).unwrap().with_device(dev);
    let hs = hl.load("hidden_states").unwrap();
    let (n, t) = (hs.dims()[0], hs.dims()[1]);
    let mask_u32: Vec<u32> = SafetensorsLoader::open(HS).unwrap().load("attention_mask").unwrap()
        .to_vec1::<i64>().unwrap().iter().map(|&x| x as u32).collect();
    let states: Vec<Tensor> = (0..n)
        .map(|i| hs.narrow(0, i, 1).unwrap().contiguous().unwrap().to_dtype(tdt).unwrap())
        .collect();

    let ref_a: Vec<f32> = SafetensorsLoader::open(AREF).unwrap().load("audio_encoding").unwrap()
        .reshape(vec![t * 2048]).unwrap().to_vec1::<f32>().unwrap();

    let cond = AudioTextConditioner::load(
        &LtxCheckpoint::open(CKPT, dev, tdt).unwrap(), dev, tdt,
    ).expect("load audio conditioner");
    let aenc = synaptix_core::grad::no_grad(|| cond.forward(&states, &mask_u32)).expect("forward");
    assert_eq!(aenc.dims(), &[1, t, 2048]);
    let ours: Vec<f32> = aenc.reshape(vec![t * 2048]).unwrap().to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();

    let h = 2048usize;
    let (mut min_cos, mut max_rel, mut worst) = (f32::INFINITY, 0.0f32, 0usize);
    for p in 0..t {
        let (mut dot, mut nr, mut no, mut dl2) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for k in 0..h {
            let r = ref_a[p * h + k] as f64;
            let o = ours[p * h + k] as f64;
            dot += r * o; nr += r * r; no += o * o; dl2 += (r - o) * (r - o);
        }
        let cos = (dot / (nr.sqrt() * no.sqrt() + 1e-12)) as f32;
        let rel = (dl2.sqrt() / (nr.sqrt() + 1e-12)) as f32;
        if cos < min_cos { min_cos = cos; worst = p; }
        if rel > max_rel { max_rel = rel; }
    }
    eprintln!("audio_encoding: per-row min_cos={min_cos:.5} (worst row {worst}) max_rel={max_rel:.4}");
    assert!(min_cos > 0.99, "per-row cos слишком низкий: {min_cos}");
}

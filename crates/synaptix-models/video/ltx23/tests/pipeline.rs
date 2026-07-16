//! Фаза 6 (капстоун): end-to-end distilled txt→video. Использует уже
//! валидированный video_encoding (Фаза 3, промпт «coffee») как контекст,
//! DiT(offload) + VAE → RGB → MP4. Проверка когерентности (finite, диапазон).
//! Гейт: SYN_LTX_GEMMA. Видео пишется в /tmp/ltx_first_video.mp4.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::vae::VaeDecoder;
use synaptix_video_ltx23::pipeline::{generate_video, rgb_to_frames};

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const CTX: &str = "tests/reference_data/ltx_gemma/textcond_video_s128.safetensors";

#[test]
fn distilled_first_video() {
    if std::env::var("SYN_LTX_GEMMA").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(CTX).exists() {
        eprintln!("skip distilled_first_video: weights/ctx absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(None);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let compute = Device::Cuda(0);

    // video_encoding (coffee prompt) [128,4096] → [1,128,4096]
    let ctx_t = SafetensorsLoader::open(CTX).unwrap().with_device(compute)
        .load("video_encoding").unwrap();
    let (ttxt, d) = (ctx_t.dims()[0], ctx_t.dims()[1]);
    let ctx = ctx_t.reshape(vec![1, ttxt, d]).unwrap();

    // ckpt на CPU → DiT offload (bf16, стримится); VAE грузится на GPU.
    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).unwrap();
    let dit = VideoDit::load_with(&ckpt, compute, DType::BF16, DType::BF16, true).expect("dit");
    let vae = VaeDecoder::load(&ckpt, compute).expect("vae");

    // малая сетка для first-light: F'=2,H'=4,W'=4 → 9 кадров 128×128.
    let (fp, hp, wp) = (2usize, 4, 4);
    let rgb = synaptix_core::grad::no_grad(|| generate_video(&dit, &vae, &ctx, fp, hp, wp, 24.0, compute))
        .expect("generate");
    eprintln!("generated rgb {:?}", rgb.dims());
    assert_eq!(rgb.dims(), &[1, 3, 9, 128, 128]);

    // когерентность: finite + разумный диапазон
    let v: Vec<f32> = rgb.reshape(vec![3 * 9 * 128 * 128]).unwrap()
        .to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let (mut mn, mut mx, mut sum, mut nan) = (f32::INFINITY, f32::NEG_INFINITY, 0.0f64, 0usize);
    for &x in &v {
        if !x.is_finite() { nan += 1; continue; }
        mn = mn.min(x); mx = mx.max(x); sum += x as f64;
    }
    let mean = sum / v.len() as f64;
    eprintln!("rgb stats: min={mn:.3} max={mx:.3} mean={mean:.3} non_finite={nan}");
    assert_eq!(nan, 0, "RGB содержит non-finite");
    assert!(mx > 0.05 && mn < -0.05 && mx - mn > 0.3, "RGB вырожден (диапазон {}..{})", mn, mx);

    // запись кадров как PPM (P6, без ffmpeg) для визуального осмотра.
    let frames = rgb_to_frames(&rgb).expect("frames");
    for (i, fr) in frames.iter().enumerate() {
        let (h, w) = (fr.dims()[1], fr.dims()[2]);
        let planar: Vec<f32> = fr.reshape(vec![3 * h * w]).unwrap().to_vec1::<f32>().unwrap();
        let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    buf.push((planar[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8);
                }
            }
        }
        std::fs::write(format!("/tmp/ltx_first_video_f{i:02}.ppm"), buf).unwrap();
    }
    eprintln!("WROTE /tmp/ltx_first_video_f*.ppm ({} кадров {}×{})", frames.len(),
        frames[0].dims()[1], frames[0].dims()[2]);
}

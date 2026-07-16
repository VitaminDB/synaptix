//! HQ: stage1 → нативный latent-upscaler ×2 → VAE decode на бóльшем разрешении
//! → PPM-кадры → ffmpeg-склейка в mp4. Гейт: SYN_LTX_HQ (тяжёлый, ~10мин).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::VideoDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::pipeline::{generate_two_stage, rgb_to_frames};
use synaptix_video_ltx23::upscaler::Upsampler;
use synaptix_video_ltx23::vae::VaeDecoder;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const UP: &str = "models/ltx2.3_v1.1/ltx-2.3-spatial-upscaler-x2-1.1.safetensors";
const CTX: &str = "tests/reference_data/ltx_gemma/textcond_video_s128.safetensors";

#[test]
fn hq_two_stage_video() {
    if std::env::var("SYN_LTX_HQ").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(UP).exists() || !Path::new(CTX).exists() {
        eprintln!("skip hq_two_stage_video: weights absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(None);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let compute = Device::Cuda(0);

    let ctx_t = SafetensorsLoader::open(CTX).unwrap().with_device(compute).load("video_encoding").unwrap();
    let (ttxt, d) = (ctx_t.dims()[0], ctx_t.dims()[1]);
    let ctx = ctx_t.reshape(vec![1, ttxt, d]).unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).unwrap();
    let dit = VideoDit::load_with(&ckpt, compute, DType::BF16, DType::BF16, true).expect("dit");
    let vae = VaeDecoder::load(&ckpt, compute).expect("vae");
    // VAE-статистики для апскейлера
    let ml = SafetensorsLoader::open(CKPT).unwrap().with_device(compute);
    let mean = ml.load("vae.per_channel_statistics.mean-of-means").unwrap();
    let std = ml.load("vae.per_channel_statistics.std-of-means").unwrap();
    let up = Upsampler::load(UP, &mean, &std, compute).expect("upscaler");

    // сетка латента из SYN_LTX_GRID="F',H',W'" (деф. 2,8,8). Выход VAE = H'·32×W'·32
    // (после upscaler ×2 удваивается). Напр. 2,16,24 → 512×768 (stage1-нативное).
    let grid = std::env::var("SYN_LTX_GRID").unwrap_or_else(|_| "2,8,8".into());
    let g: Vec<usize> = grid.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let (fp, hp, wp) = (g[0], g[1], g[2]);
    // SYN_LTX_NOUP=1 → без upscaler (декод stage1 напрямую).
    let no_up = std::env::var("SYN_LTX_NOUP").is_ok();
    let stage2 = std::env::var("SYN_LTX_STAGE2").is_ok();
    let rgb = synaptix_core::grad::no_grad(|| -> Result<_, _> {
        if no_up {
            synaptix_video_ltx23::pipeline::generate_video(&dit, &vae, &ctx, fp, hp, wp, 24.0, compute)
        } else {
            generate_two_stage(&dit, &up, &vae, &ctx, fp, hp, wp, 24.0, stage2, compute)
        }
    })
    .expect("generate");
    eprintln!("HQ rgb {:?} (grid {grid}, no_up={no_up}, stage2={stage2})", rgb.dims());

    let frames = rgb_to_frames(&rgb).expect("frames");
    let dir = "/tmp/ltx_hq_frames";
    std::fs::create_dir_all(dir).unwrap();
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
        std::fs::write(format!("{dir}/f{i:03}.ppm"), buf).unwrap();
    }
    // ffmpeg-склейка
    let out = "/tmp/ltx_hq.mp4";
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-framerate", "12", "-i", &format!("{dir}/f%03d.ppm"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", out])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("WROTE {out} ({} кадров {}×{})", frames.len(),
            frames[0].dims()[1], frames[0].dims()[2]),
        other => eprintln!("ffmpeg: {other:?} (PPM в {dir})"),
    }
}

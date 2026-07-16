//! Фаза 11: end-to-end совместная генерация видео+аудио → mp4 с дорожкой.
//! Гейт SYN_LTX_AV (тяжёлый, полный 48-блочный offload). Использует сохранённые
//! video/audio encoding одного промпта (textcond_*_s128).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::audio_vae::AudioVaeDecoder;
use synaptix_video_ltx23::dit::AvDit;
use synaptix_video_ltx23::loader::LtxCheckpoint;
use synaptix_video_ltx23::pipeline::{generate_av, rgb_to_frames};
use synaptix_video_ltx23::vae::VaeDecoder;
use synaptix_video_ltx23::vocoder::VocoderWithBwe;

const CKPT: &str = "models/ltx2.3_v1.1/ltx-2.3-22b-distilled-1.1.safetensors";
const VENC: &str = "tests/reference_data/ltx_gemma/textcond_video_s128.safetensors";
const AENC: &str = "tests/reference_data/ltx_gemma/textcond_audio_s128.safetensors";

fn write_wav(path: &str, wave: &Tensor, sr: u32) {
    // wave [1,2,L] f32 [-1,1] → interleaved PCM16 stereo WAV.
    let (ch, l) = (wave.dims()[1], wave.dims()[2]);
    let data: Vec<f32> = wave.contiguous().unwrap().reshape(vec![ch * l]).unwrap()
        .to_dtype(DType::F32).unwrap().to_vec1::<f32>().unwrap();
    let mut pcm: Vec<u8> = Vec::with_capacity(l * ch * 2);
    for i in 0..l {
        for c in 0..ch {
            let s = (data[c * l + i].clamp(-1.0, 1.0) * 32767.0) as i16;
            pcm.extend_from_slice(&s.to_le_bytes());
        }
    }
    let byte_rate = sr * ch as u32 * 2;
    let block_align = (ch * 2) as u16;
    let data_len = pcm.len() as u32;
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&(36 + data_len).to_le_bytes());
    buf.extend_from_slice(b"WAVEfmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
    buf.extend_from_slice(&(ch as u16).to_le_bytes());
    buf.extend_from_slice(&sr.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&16u16.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());
    buf.extend_from_slice(&pcm);
    std::fs::write(path, buf).unwrap();
}

#[test]
fn av_generate_and_mux() {
    if std::env::var("SYN_LTX_AV").is_err() {
        return;
    }
    if !Path::new(CKPT).exists() || !Path::new(VENC).exists() || !Path::new(AENC).exists() {
        eprintln!("skip av_generate_and_mux: weights/encodings absent");
        return;
    }
    synaptix_video_ltx23::runtime::set_dit_nblocks_cap(None);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);

    let venc = SafetensorsLoader::open(VENC).unwrap().with_device(dev).load("video_encoding").unwrap();
    let (tv, dv) = (venc.dims()[0], venc.dims()[1]);
    let venc = venc.reshape(vec![1, tv, dv]).unwrap();
    let aenc = SafetensorsLoader::open(AENC).unwrap().with_device(dev).load("audio_encoding").unwrap();
    let (ta, da) = (aenc.dims()[0], aenc.dims()[1]);
    let aenc = aenc.reshape(vec![1, ta, da]).unwrap();

    let ckpt = LtxCheckpoint::open(CKPT, Device::Cpu, DType::BF16).unwrap();
    let dit = AvDit::load_with(&ckpt, dev, DType::BF16, DType::BF16, true).expect("avdit");
    let vae = VaeDecoder::load(&ckpt, dev).expect("vae");
    let audio_vae = AudioVaeDecoder::load(&ckpt, dev).expect("audio_vae");
    let vocoder = VocoderWithBwe::load(CKPT, dev).expect("vocoder");

    let grid = std::env::var("SYN_LTX_GRID").unwrap_or_else(|_| "2,8,8".into());
    let g: Vec<usize> = grid.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let (fp, hp, wp) = (g[0], g[1], g[2]);
    let fps = 24.0;

    let (rgb, wave) = synaptix_core::grad::no_grad(|| {
        generate_av(&dit, &vae, &audio_vae, &vocoder, &venc, &aenc, fp, hp, wp, fps, dev)
    }).expect("generate_av");
    eprintln!("rgb {:?}  wave {:?}", rgb.dims(), wave.dims());

    // кадры → PPM
    let frames = rgb_to_frames(&rgb).expect("frames");
    let dir = "/tmp/ltx_av_frames";
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
    write_wav("/tmp/ltx_av.wav", &wave, 48000);

    // ffmpeg: кадры + аудио → mp4
    let out = "/tmp/ltx_av.mp4";
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-framerate", "24", "-i", &format!("{dir}/f%03d.ppm"),
               "-i", "/tmp/ltx_av.wav", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", out])
        .status();
    match status {
        Ok(s) if s.success() => eprintln!("WROTE {out} ({} кадров {}×{}, аудио {} сэмплов)",
            frames.len(), frames[0].dims()[1], frames[0].dims()[2], wave.dims()[2]),
        other => eprintln!("ffmpeg: {other:?} (PPM в {dir}, wav /tmp/ltx_av.wav)"),
    }
    assert!(wave.dims()[2] > 0);
}

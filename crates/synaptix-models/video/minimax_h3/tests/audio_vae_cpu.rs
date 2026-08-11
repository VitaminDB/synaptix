use std::path::PathBuf;
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3::audio_vae::{interleave_stereo, AudioVae};
use synaptix_video_minimax_h3::config::AudioVaeConfig;
use synaptix_video_minimax_h3::loader::{ComponentLoader, H3Paths};

fn model_dir() -> Option<PathBuf> {
    let p = std::env::var("H3_MODEL_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(std::env::var("HOME").unwrap_or_default())
            .join(".local/share/synthos/hf/MiniMax-H3")
    });
    (p.join("FL2VA").is_dir() || p.join("transformer").is_dir()).then_some(p)
}

#[test]
fn decoder_produces_stereo_waveform_of_expected_length() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = model_dir() else { return };
    let paths = H3Paths::open(&dir).expect("paths");
    let cfg = AudioVaeConfig::from_dir(&paths.root).expect("config");
    let hop = cfg.hop_length();

    let t0 = Instant::now();
    let w = ComponentLoader::open_file(paths.audio_vae_file(), Device::Cpu).expect("weights");
    let vae = AudioVae::load_decoder(&w, cfg, Device::Cpu, DType::F32).expect("load decoder");
    eprintln!("[audio-cpu] загрузка за {:.1} с", t0.elapsed().as_secs_f32());

    let latent_t = 3usize;
    let n = 32 * 2 * latent_t;
    let data: Vec<f32> = (0..n).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let latent = Tensor::from_vec(data, vec![1, 32, 2, latent_t], Device::Cpu).expect("latent");

    let t1 = Instant::now();
    let wave = vae.decode(&latent).expect("decode");
    eprintln!(
        "[audio-cpu] декод {latent_t} латент-кадров за {:.1} с",
        t1.elapsed().as_secs_f32()
    );

    assert_eq!(wave.dims()[0], 1, "batch");
    assert_eq!(wave.dims()[1], 2, "стерео-каналы");
    assert_eq!(wave.dims()[2], latent_t * hop, "{hop} сэмплов на латентный кадр");

    let pcm = interleave_stereo(&wave).expect("interleave");
    assert_eq!(pcm.len(), 2 * latent_t * hop);
    assert!(pcm.iter().all(|v| v.is_finite()), "в waveform есть NaN/Inf");
    assert!(
        pcm.iter().all(|v| (-1.0..=1.0).contains(v)),
        "BigVGAN обязан клампить выход в [-1, 1]"
    );
    let energy = pcm.iter().map(|v| v * v).sum::<f32>() / pcm.len() as f32;
    assert!(energy > 0.0, "декодер вернул тишину — веса не применились");
    eprintln!("[audio-cpu] RMS выхода {:.4}", energy.sqrt());
}

#[test]
fn channels_are_decoded_independently() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = model_dir() else { return };
    let paths = H3Paths::open(&dir).expect("paths");
    let cfg = AudioVaeConfig::from_dir(&paths.root).expect("config");
    let hop = cfg.hop_length();
    let w = ComponentLoader::open_file(paths.audio_vae_file(), Device::Cpu).expect("weights");
    let vae = AudioVae::load_decoder(&w, cfg, Device::Cpu, DType::F32).expect("load decoder");

    let latent_t = 2usize;
    let mut data = vec![0f32; 32 * 2 * latent_t];
    for c in 0..32 {
        for t in 0..latent_t {
            data[(c * 2) * latent_t + t] = 0.3;
            data[(c * 2 + 1) * latent_t + t] = -0.3;
        }
    }
    let latent = Tensor::from_vec(data, vec![1, 32, 2, latent_t], Device::Cpu).expect("latent");
    let wave = vae.decode(&latent).expect("decode");
    let flat = wave
        .reshape(vec![2 * latent_t * hop])
        .and_then(|t| t.to_vec1::<f32>())
        .expect("to vec");
    let half = latent_t * hop;
    let l_rms = (flat[..half].iter().map(|v| v * v).sum::<f32>() / half as f32).sqrt();
    let r_rms = (flat[half..].iter().map(|v| v * v).sum::<f32>() / half as f32).sqrt();
    eprintln!("[audio-cpu] L RMS {l_rms:.4} · R RMS {r_rms:.4}");
    assert!(l_rms > 0.0 && r_rms > 0.0);
    assert!(
        (l_rms - r_rms).abs() > 1e-6,
        "каналы с разным латентом дали одинаковый выход — стерео схлопнулось в моно"
    );
}

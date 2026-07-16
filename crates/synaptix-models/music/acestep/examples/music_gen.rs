//! e2e ACE-Step text-to-music на CUDA.
//! Запуск: cargo run --release --features cuda -p synaptix-music-acestep --example music_gen -- "calm ambient piano" 8

use std::path::{Path, PathBuf};

use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_acestep::ar::CodesGenOptions;
use synaptix_music_acestep::pipeline::{generate_music, MusicPaths, SamplerOptions};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let dir = Path::new("storage/syn_models");
    let lm: PathBuf = dir.join("acestep_5hz_lm_1.7b.syn");
    let te: PathBuf = dir.join("qwen3-embedding-0.6b.syn");
    let dit: PathBuf = dir.join("acestep_v15_xl_base.syn");
    let vae: PathBuf = dir.join("acestep_vae.syn");
    let paths = MusicPaths { lm: &lm, text_encoder: &te, dit: &dit, vae: &vae };

    let caption = std::env::args().nth(1).unwrap_or_else(|| "calm ambient piano".to_string());
    let dur: u32 = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let gscale: f32 = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(7.0);
    let steps: usize = std::env::args().nth(5).and_then(|s| s.parse().ok()).unwrap_or(32);
    let opts = SamplerOptions { steps, shift: 3.0, guidance_scale: gscale, ..Default::default() };

    eprintln!("[music] caption={caption:?} dur={dur}s guidance={gscale} steps={}", opts.steps);
    let t0 = std::time::Instant::now();
    let use_cot = std::env::args().nth(4).map(|s| s == "cot" || s == "1").unwrap_or(false);
    let copts = CodesGenOptions { seed: 42, top_k: 0, cfg_scale: 2.0, ..CodesGenOptions::default() };
    let edit = synaptix_music_acestep::pipeline::EditOptions::default();
    let extras = synaptix_music_acestep::pipeline::GenExtras::ar_on();
    let (samples, sr, _latent) =
        generate_music(&paths, &caption, "", dur, Device::Cuda(0), DType::F32, DType::F32, DType::F32, &opts, &copts, use_cot, &edit, &extras)
            .expect("generate_music");
    let out = "/tmp/acestep_out.wav";
    synaptix_audio::write_wav_mono_f32(out, &samples, sr).expect("write wav");
    eprintln!(
        "[music] wrote {} samples ({:.1}s) to {out} in {:.1}s",
        samples.len(),
        samples.len() as f32 / sr as f32,
        t0.elapsed().as_secs_f32()
    );
}

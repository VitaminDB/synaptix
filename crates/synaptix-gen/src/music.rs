//! Музыка одним вызовом: ACE-Step v1.5 (музыка по описанию и тексту) и YuE2
//! (песня с вокалом). Умолчания — как у `synaptix music` / `synaptix song`.
//! Отмена — флагом `cancel`; прогресс — `progress(готово, всего)`, где он есть.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

type R<T> = Result<T, String>;

#[derive(Debug, Clone, Default)]
pub struct MusicRequest {
    /// Описание: жанр, инструменты, настроение (ACE-Step — caption, YuE2 — style).
    pub prompt: String,
    /// Текст песни (с метками [verse]/[chorus]); пусто — инструментал.
    pub lyrics: String,
    /// Длительность, секунды; 0 — модель решает сама.
    pub seconds: u32,
    pub seed: u64,
    pub steps: Option<usize>,
    /// CFG (ACE-Step); `None` — умолчание варианта.
    pub guidance: Option<f32>,
    /// Квант: `nvfp4`, `mxfp8`, `bf16`; пусто/`auto` — плотный.
    pub quant: Option<String>,
}

pub struct MusicOutput {
    /// Отсчёты, перемежённые по каналам.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub label: String,
}

/// Компоненты ACE-Step: DiT (сама модель), LM кодов, текстовый энкодер, VAE.
pub struct AcePaths {
    pub dit: PathBuf,
    pub lm: PathBuf,
    pub text_encoder: PathBuf,
    pub vae: PathBuf,
}

pub fn acestep(paths: &AcePaths, device: Device, req: &MusicRequest, cancel: Arc<AtomicBool>) -> R<MusicOutput> {
    use synaptix_music_acestep::ar::CodesGenOptions;
    use synaptix_music_acestep::pipeline::{generate_music, EditOptions, GenExtras, MusicPaths, NormMode, SamplerOptions};
    use synaptix_music_acestep::DitVariant;
    for (label, p) in [("DiT", &paths.dit), ("LM", &paths.lm), ("текстовый энкодер", &paths.text_encoder), ("VAE", &paths.vae)] {
        if !p.exists() {
            return Err(format!("ACE-Step: нет компонента {label}: {}", p.display()));
        }
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let variant = DitVariant::detect(&paths.dit);
    let steps = req.steps.filter(|&s| s > 0).unwrap_or(variant.default_steps());
    let opts = SamplerOptions { steps, shift: variant.default_shift(), guidance_scale: req.guidance.unwrap_or(variant.default_cfg()), ..Default::default() };
    let copts = CodesGenOptions { seed: req.seed.max(1), ..CodesGenOptions::default() };
    let extras = GenExtras { norm_mode: NormMode::Peak, ..Default::default() };
    let mp = MusicPaths { lm: &paths.lm, text_encoder: &paths.text_encoder, dit: &paths.dit, vae: &paths.vae };
    let compute = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let quant = match req.quant.as_deref().unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4" if device.is_cuda() => DType::NVFP4,
        "mxfp8" | "fp8" if device.is_cuda() => DType::MXFP8,
        _ => compute,
    };
    let (samples, sample_rate, _latent) = synaptix_music_acestep::with_cancel(Some(cancel), || {
        generate_music(&mp, &req.prompt, &req.lyrics, req.seconds, device, compute, quant, quant, &opts, &copts, true, &EditOptions::default(), &extras, None)
    })
    .map_err(|e| e.to_string())?;
    let secs = samples.len() as f32 / sample_rate.max(1) as f32;
    Ok(MusicOutput { samples, sample_rate, channels: 1, label: format!("ACE-Step {variant:?}, {secs:.0} с, {steps} шагов") })
}

/// YuE2: песня (стиль + текст). `model` и `vae` — бандлы.
pub fn yue2(model: &Path, vae: &Path, device: Device, req: &MusicRequest, cancel: Arc<AtomicBool>, progress: &mut dyn FnMut(usize, usize)) -> R<MusicOutput> {
    use synaptix_music_yue2::pipeline::{Callbacks, Yue2Options, Yue2Paths, Yue2Pipeline};
    use synaptix_music_yue2::protocol::{seconds_to_frames, Cot, GenerationConfig, SongRequest, CONTEXT};
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mut generation = GenerationConfig::default();
    if let Some(s) = req.steps.filter(|&s| s > 0) {
        generation.ode_steps = s;
    }
    let seconds = if req.seconds > 0 { req.seconds } else { 60 };
    generation.semantic.max_tokens = seconds_to_frames(seconds as f32).clamp(1, CONTEXT - 2);
    let options = Yue2Options {
        device,
        compute: DType::BF16,
        quant: match req.quant.as_deref().unwrap_or("").to_ascii_lowercase().as_str() {
            "nvfp4" => Some(DType::NVFP4),
            "mxfp8" | "fp8" => Some(DType::MXFP8),
            _ => None,
        },
        vae_dtype: DType::F32,
        vae_core_frames: 1024,
        vae_halo_frames: 16,
        generation,
    };
    let mut pipe = Yue2Pipeline::open(&Yue2Paths { model: model.to_path_buf(), vae: vae.to_path_buf() }, options).map_err(|e| e.to_string())?;
    let request = SongRequest { style: req.prompt.clone(), lyrics: req.lyrics.clone(), cot: Cot::Full, seed: if req.seed == 0 { 831001 } else { req.seed }, abc: None, cfg_scale: None };
    let progress = std::cell::RefCell::new(progress);
    let on_ode = |done: usize, total: usize| (progress.borrow_mut())(done, total);
    let is_cancelled = || cancel.load(std::sync::atomic::Ordering::Relaxed);
    let cb = Callbacks { cancel: Some(&is_cancelled), on_token: None, on_ode: Some(&on_ode), on_vae: None };
    let song = pipe.run(&request, cb).map_err(|e| e.to_string())?;
    let secs = song.audio.len() as f32 / (song.sample_rate.max(1) as f32 * song.channels.max(1) as f32);
    Ok(MusicOutput { samples: song.audio, sample_rate: song.sample_rate, channels: song.channels as u16, label: format!("YuE2, {secs:.0} с") })
}

/// WAV 16 бит из перемежённых отсчётов.
pub fn wav_bytes(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + samples.len() * 2);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + data_len).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&(sample_rate * channels as u32 * 2).to_le_bytes());
    w.extend_from_slice(&(channels * 2).to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());
    for &x in samples {
        w.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    w
}

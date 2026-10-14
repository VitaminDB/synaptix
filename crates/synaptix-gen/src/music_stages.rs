//! Стадии музыки для графа службы: ACE-Step (модель → [латент записи] →
//! генерация во всех режимах) и YuE2 (модель → песня с партитурой и
//! латентом → декод; транскрипция записи в партитуру SheetSage2).
//! Параметры и умолчания — как у прежних родных нод Synthos. Латенты
//! хранятся в памяти хоста: между стадиями видеокарта свободна.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::music::AcePaths;

type R<T> = Result<T, String>;

fn kernels() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
}

fn on_host(t: &Tensor) -> R<Tensor> {
    t.to_device(Device::Cpu).and_then(|t| t.to_dtype(DType::F32)).map_err(|e| e.to_string())
}

/// Запись в каналы заданной частоты (ffmpeg).
fn channels(path: &Path, rate: u32, ch: usize) -> R<Vec<Vec<f32>>> {
    let pcm = crate::video::ffmpeg_f32(path, rate, ch as u32)?;
    if pcm.is_empty() {
        return Err(format!("{}: пустая запись", path.display()));
    }
    Ok((0..ch).map(|c| pcm.iter().skip(c).step_by(ch).copied().collect()).collect())
}

// ── ACE-Step ──────────────────────────────────────────────────────────────

/// Режимы генерации ACE-Step (порядок — как у прежней ноды).
pub const ACE_MODES: &[&str] = &["text2music", "retake", "repaint", "extend", "edit", "cover", "extract"];
/// Тональность: пусто — LM выводит сама.
pub const KEYSCALES: &[&str] = &[
    "", "C major", "C minor", "C# major", "C# minor", "D major", "D minor", "D# major", "D# minor", "E major", "E minor", "F major", "F minor", "F# major", "F# minor", "G major",
    "G minor", "G# major", "G# minor", "A major", "A minor", "A# major", "A# minor", "B major", "B minor",
];
/// Размер такта: пусто — LM выводит сама.
pub const TIMESIGS: &[&str] = &["", "4/4", "3/4", "6/8", "2/4", "5/4", "7/8", "9/8", "12/8"];
/// Дорожки режима extract.
pub const TRACKS: &[&str] = &["vocals", "backing_vocals", "drums", "bass", "guitar", "keyboard", "percussion", "strings", "synth", "fx", "brass", "woodwinds"];
pub const NORMS: &[&str] = &["peak", "rms", "off"];
pub const DCW_MODES: &[&str] = &["off", "double", "low", "high", "pix"];
/// LM кодов: полная (4B) или малая (1.7B).
pub const ACE_LMS: &[&str] = &["4b", "1.7b"];

pub struct AceModel {
    paths: AcePaths,
    device: Device,
    compute: DType,
    quant: DType,
    variant: synaptix_music_acestep::DitVariant,
}

/// Латент записи или прошлой генерации: `[1, 64, кадры]`, 25 кадров в секунду.
pub struct AceLatent {
    pub latent: Tensor,
    pub frames: usize,
}

impl AceLatent {
    fn new(t: &Tensor) -> R<Self> {
        let t = on_host(t)?;
        let d = t.dims().to_vec();
        let t = match d.as_slice() {
            [1, 64, _] => t,
            [1, _, 64] => t.transpose(1, 2).and_then(|t| t.contiguous()).map_err(|e| e.to_string())?,
            other => return Err(format!("латент ACE-Step формы {other:?}")),
        };
        let frames = t.dims()[2];
        Ok(Self { latent: t, frames })
    }

    pub fn seconds(&self) -> f32 {
        self.frames as f32 / 25.0
    }
}

/// Запрос генерации. Нули в `steps`/`guidance`/`shift` — умолчания варианта DiT.
#[derive(Debug, Clone)]
pub struct AceRequest {
    pub mode: String,
    pub caption: String,
    pub lyrics: String,
    /// ≤ 0 — длину выбирает LM (в режимах по записи — длина записи).
    pub seconds: f32,
    pub steps: usize,
    pub guidance: f32,
    pub shift: f32,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub lm_guidance: f32,
    pub use_cot: bool,
    pub use_ar: bool,
    pub bpm: u32,
    pub keyscale: String,
    pub timesig: String,
    pub norm: String,
    pub dcw: String,
    pub dcw_scaler: f32,
    pub dcw_high_scaler: f32,
    pub retake_variance: f32,
    pub retake_seed: u64,
    pub repaint_start: f32,
    pub repaint_end: f32,
    pub repaint_strength: f32,
    pub edit_n_min: f32,
    pub edit_n_max: f32,
    pub track: String,
}

impl Default for AceRequest {
    fn default() -> Self {
        Self {
            mode: "text2music".into(),
            caption: String::new(),
            lyrics: String::new(),
            seconds: -1.0,
            steps: 0,
            guidance: 0.0,
            shift: 0.0,
            seed: 1,
            temperature: 0.85,
            top_p: 0.9,
            top_k: 0,
            min_p: 0.0,
            lm_guidance: 2.0,
            use_cot: true,
            use_ar: true,
            bpm: 0,
            keyscale: String::new(),
            timesig: String::new(),
            norm: "peak".into(),
            dcw: "off".into(),
            dcw_scaler: 0.02,
            dcw_high_scaler: 0.06,
            retake_variance: 0.5,
            retake_seed: 1,
            repaint_start: 0.0,
            repaint_end: -1.0,
            repaint_strength: 0.7,
            edit_n_min: 0.0,
            edit_n_max: 1.0,
            track: "vocals".into(),
        }
    }
}

pub struct AceOutput {
    /// Моно 48 кГц.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub latent: AceLatent,
    pub label: String,
}

/// ACE-Step: DiT и компоненты, устройство, квант (пусто — плотно).
pub fn ace_open(paths: AcePaths, device: Device, quant: Option<&str>) -> R<AceModel> {
    for (label, p) in [("DiT", &paths.dit), ("LM", &paths.lm), ("текстовый энкодер", &paths.text_encoder), ("VAE", &paths.vae)] {
        if !p.exists() {
            return Err(format!("ACE-Step: нет компонента {label}: {}", p.display()));
        }
    }
    let compute = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let quant = match quant.unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4" if device.is_cuda() => DType::NVFP4,
        "mxfp8" | "fp8" if device.is_cuda() => DType::MXFP8,
        _ => compute,
    };
    let variant = synaptix_music_acestep::DitVariant::detect(&paths.dit);
    Ok(AceModel { paths, device, compute, quant, variant })
}

impl AceModel {
    pub fn name(&self) -> String {
        let lm = self.paths.lm.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        format!("ACE-Step {:?}, LM {lm}", self.variant)
    }

    /// Запись → латент (стерео 48 кГц; окна по `chunk` с, контекст `overlap` с).
    pub fn encode(&self, audio: &Path, chunk: f32, overlap: f32, cancel: &dyn Fn() -> bool) -> R<AceLatent> {
        use synaptix_music_acestep::vae::AceStepVae;
        kernels();
        let ch = channels(audio, 48_000, 2)?;
        let n = ch[0].len().min(ch[1].len());
        // Громкий мастер не должен выходить за [-1, 1] — как у ACE-Step.
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend(ch[0][..n].iter().map(|x| x.clamp(-1.0, 1.0)));
        flat.extend(ch[1][..n].iter().map(|x| x.clamp(-1.0, 1.0)));
        let t = Tensor::from_vec(flat, vec![1, 2, n], self.device).map_err(|e| e.to_string())?;
        let vae = AceStepVae::open(&self.paths.vae, self.device).map_err(|e| e.to_string())?;
        let chunk_frames = ((chunk.max(1.0)) * 25.0).round() as usize;
        let overlap_frames = ((overlap.max(0.0)) * 25.0).round() as usize;
        let lat = vae.encode_mean_tiled(&t, chunk_frames, overlap_frames, cancel).map_err(|e| e.to_string())?;
        AceLatent::new(&lat)
    }

    pub fn generate(&self, req: &AceRequest, src: Option<&AceLatent>, cancel: Arc<AtomicBool>) -> R<AceOutput> {
        use synaptix_music_acestep::ar::CodesGenOptions;
        use synaptix_music_acestep::dcw::{DcwCorrector, DcwMode};
        use synaptix_music_acestep::pipeline::{generate_music, EditMode, EditOptions, GenExtras, MusicPaths, NormMode, SamplerOptions};
        use synaptix_music_acestep::DitVariant;
        kernels();
        let mode = ACE_MODES.iter().position(|m| *m == req.mode).unwrap_or(0);
        let edit_mode = match mode {
            1 => EditMode::Retake,
            2 | 3 => EditMode::Repaint,
            4 => EditMode::Edit,
            5 => EditMode::Cover,
            6 => EditMode::Extract,
            _ => EditMode::Text2Music,
        };
        // Retake варьирует шум и без записи.
        if mode >= 2 && src.is_none() {
            return Err(format!("режиму {} нужен латент записи или прошлой генерации", ACE_MODES[mode]));
        }
        // Без LM кодов штатно работает только turbo; extract LM не зовёт.
        if mode != 6 && !req.use_ar && self.variant != DitVariant::Turbo {
            return Err("без LM кодов (ar = выкл.) работает только turbo-вариант DiT".into());
        }
        let steps = if req.steps > 0 { req.steps } else { self.variant.default_steps() };
        let guidance = if req.guidance > 0.0 { req.guidance } else { self.variant.default_cfg() };
        let shift = if req.shift > 0.0 { req.shift } else { self.variant.default_shift() };
        let dcw = match req.dcw.as_str() {
            "off" | "" => DcwCorrector { enabled: false, ..Default::default() },
            m => DcwCorrector {
                enabled: true,
                mode: match m {
                    "low" => DcwMode::Low,
                    "high" => DcwMode::High,
                    "pix" => DcwMode::Pix,
                    _ => DcwMode::Double,
                },
                scaler: req.dcw_scaler,
                high_scaler: req.dcw_high_scaler,
            },
        };
        let opts = SamplerOptions { steps, shift, guidance_scale: guidance, dcw };
        let copts = CodesGenOptions {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            min_p: req.min_p,
            cfg_scale: req.lm_guidance,
            seed: req.seed.max(1),
            ..CodesGenOptions::default()
        };
        let src_cl = match src {
            Some(s) => Some(
                s.latent
                    .to_device(self.device)
                    .and_then(|t| t.transpose(1, 2))
                    .and_then(|t| t.contiguous())
                    .map_err(|e| e.to_string())?,
            ),
            None => None,
        };
        // Режимы по записи (кроме extend) — длина записи.
        let seconds = match (mode, src) {
            (2 | 4 | 5 | 6, Some(s)) => (s.seconds().round() as u32).max(1),
            _ if req.seconds > 0.0 => req.seconds.round() as u32,
            _ => 0,
        };
        let edit = EditOptions {
            mode: edit_mode,
            track_name: if TRACKS.contains(&req.track.as_str()) { req.track.clone() } else { TRACKS[0].into() },
            retake_variance: req.retake_variance,
            retake_seed: req.retake_seed.max(1),
            src_latent: src_cl,
            repaint_start_sec: req.repaint_start,
            repaint_end_sec: req.repaint_end,
            repaint_strength: req.repaint_strength,
            edit_n_min: req.edit_n_min,
            edit_n_max: req.edit_n_max,
            edit_n_avg: 1,
            edit_source_caption: String::new(),
            edit_source_lyric: String::new(),
        };
        let some = |s: &str| (!s.trim().is_empty()).then(|| s.trim().to_string());
        let extras = GenExtras {
            use_ar: req.use_ar,
            bpm: (req.bpm > 0).then_some(req.bpm),
            keyscale: some(&req.keyscale),
            timesig: some(&req.timesig),
            norm_mode: match req.norm.as_str() {
                "rms" => NormMode::Rms,
                "off" => NormMode::Off,
                _ => NormMode::Peak,
            },
            on_stage: None,
        };
        let p = &self.paths;
        let mp = MusicPaths { lm: &p.lm, text_encoder: &p.text_encoder, dit: &p.dit, vae: &p.vae };
        let (samples, sample_rate, latent) = synaptix_music_acestep::with_cancel(Some(cancel), || {
            generate_music(&mp, &req.caption, &req.lyrics, seconds, self.device, self.compute, self.quant, self.quant, &opts, &copts, req.use_cot, &edit, &extras, None)
        })
        .map_err(|e| e.to_string())?;
        let latent = AceLatent::new(&latent)?;
        let secs = samples.len() as f32 / sample_rate.max(1) as f32;
        let label = format!("ACE-Step {}, {secs:.0} с, {steps} шагов, сид {}", ACE_MODES[mode], copts.seed);
        Ok(AceOutput { samples, sample_rate, latent, label })
    }
}

// ── YuE2 ──────────────────────────────────────────────────────────────────

/// Партитура: с аккордами, только мелодия, без партитуры.
pub const COTS: &[&str] = &["full", "melody", "off"];
pub const TRANSCRIBE_MODES: &[&str] = &["melody", "full"];
pub const VOICES: &[&str] = &["both", "vocal", "ins"];

pub struct YueModel {
    model: PathBuf,
    vae: PathBuf,
    device: Device,
    quant: Option<DType>,
}

/// Латенты акустики YuE2 (`[1, 64, кадры]`, на хосте).
pub struct YueLatent {
    pub latent: Tensor,
    pub seconds: f32,
}

#[derive(Debug, Clone)]
pub struct YueRequest {
    pub style: String,
    pub lyrics: String,
    /// Готовая партитура (сильнее сгенерированной; при `cot = off` не нужна).
    pub abc: Option<String>,
    pub cot: String,
    /// 0 — пока модель не закончит сама.
    pub seconds: f32,
    pub steps: usize,
    /// 0 — умолчание режима.
    pub guidance: f32,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
}

impl Default for YueRequest {
    fn default() -> Self {
        Self {
            style: String::new(),
            lyrics: String::new(),
            abc: None,
            cot: "full".into(),
            seconds: 60.0,
            steps: 32,
            guidance: 0.0,
            seed: 1,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 100,
            repetition_penalty: 1.2,
        }
    }
}

pub struct YueSong {
    pub score: Option<String>,
    pub latent: YueLatent,
    pub label: String,
}

pub fn yue_open(model: &Path, vae: &Path, device: Device, quant: Option<&str>) -> R<YueModel> {
    for (label, p) in [("модель", model), ("декодер", vae)] {
        if !p.exists() {
            return Err(format!("YuE2: нет файла ({label}): {}", p.display()));
        }
    }
    let quant = match quant.unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4" if device.is_cuda() => Some(DType::NVFP4),
        "mxfp8" | "fp8" if device.is_cuda() => Some(DType::MXFP8),
        _ => None,
    };
    Ok(YueModel { model: model.to_path_buf(), vae: vae.to_path_buf(), device, quant })
}

impl YueModel {
    pub fn name(&self) -> String {
        let n = self.model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        format!("YuE2 {n}")
    }

    /// Партитура и акустика. `progress(доля 0..1)`.
    pub fn generate(&self, req: &YueRequest, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> R<YueSong> {
        use synaptix_music_yue2::ar::Phase;
        use synaptix_music_yue2::protocol::{seconds_to_frames, Cot, GenerationConfig, SongRequest, CONTEXT};
        use synaptix_music_yue2::{Callbacks, Yue2Options, Yue2Paths, Yue2Pipeline};
        kernels();
        let cot = match req.cot.as_str() {
            "melody" => Cot::Melody,
            "off" => Cot::Off,
            _ => Cot::Full,
        };
        let request = SongRequest {
            style: req.style.clone(),
            lyrics: req.lyrics.clone(),
            cot,
            seed: req.seed.max(1),
            abc: if cot == Cot::Off { None } else { req.abc.clone().filter(|a| !a.trim().is_empty()) },
            cfg_scale: (req.guidance > 0.0).then_some(req.guidance),
        };
        request.validate().map_err(|e| e.to_string())?;
        let mut generation = GenerationConfig::default();
        generation.ode_steps = req.steps.max(1);
        if req.seconds > 0.0 {
            let frames = seconds_to_frames(req.seconds).clamp(1, CONTEXT - 2);
            generation.semantic.max_tokens = frames;
            generation.semantic.min_tokens = generation.semantic.min_tokens.min(frames);
        }
        generation.semantic.temperature = req.temperature;
        generation.semantic.top_p = req.top_p;
        generation.semantic.top_k = req.top_k.max(1);
        generation.semantic.repetition_penalty = req.repetition_penalty;
        generation.semantic.validate().map_err(|e| e.to_string())?;
        let abc_budget = generation.abc.max_tokens.max(1) as f32;
        let sem_budget = generation.semantic.max_tokens.max(1) as f32;
        let options = Yue2Options { device: self.device, compute: DType::BF16, quant: self.quant, generation, ..Default::default() };
        let pipe = Yue2Pipeline::open(&Yue2Paths { model: self.model.clone(), vae: self.vae.clone() }, options).map_err(|e| e.to_string())?;
        // Доли: партитура 15 %, музыка 50 %, акустика 35 %.
        let on_token = |phase: Phase, _t: u32, step: usize| match phase {
            Phase::Abc => progress(0.15 * (step as f32 / abc_budget).min(1.0)),
            Phase::Semantic => progress(0.15 + 0.5 * (step as f32 / sem_budget).min(1.0)),
        };
        let on_ode = |done: usize, total: usize| progress(0.65 + 0.35 * if total == 0 { 0.0 } else { (done as f32 / total as f32).min(1.0) });
        let cb = Callbacks { cancel: Some(cancel), on_token: Some(&on_token), on_ode: Some(&on_ode), on_vae: None };
        let semantic = pipe.generate(&request, cb).map_err(|e| e.to_string())?;
        let latents = pipe.synthesize(&semantic, cb).map_err(|e| e.to_string())?;
        let latent = yue_latent(&latents, semantic.seconds())?;
        let label = format!("YuE2, {:.0} с, партитура {} ток., сид {}", latent.seconds, semantic.plan.abc_ids.len(), request.seed);
        Ok(YueSong { score: semantic.plan.abc.clone(), latent, label })
    }

    /// Латенты → перемежённое стерео 48 кГц.
    pub fn decode(&self, lat: &YueLatent, core_frames: usize, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> R<Vec<f32>> {
        use synaptix_music_yue2::vae::Yue2Vae;
        kernels();
        let vae = Yue2Vae::open(&self.vae, self.device, DType::F32, true).map_err(|e| e.to_string())?;
        let z = lat.latent.to_device(self.device).map_err(|e| e.to_string())?;
        let on = |done: usize, total: usize| progress(if total == 0 { 0.0 } else { (done as f32 / total as f32).min(1.0) });
        let audio = vae.decode_tiled(&z, core_frames.max(16), 16, cancel, &on).map_err(|e| e.to_string())?;
        synaptix_music_yue2::pipeline::interleave(&audio).map_err(|e| e.to_string())
    }

    /// Запись → партитура ABC (SheetSage2 из того же бандла).
    pub fn transcribe(&self, audio: &Path, mode: &str, voices: &str, cancel: &dyn Fn() -> bool, progress: &dyn Fn(f32)) -> R<String> {
        use synaptix_music_sheetsage2::pipeline::{prepare_audio, Progress};
        use synaptix_music_sheetsage2::{SheetSage2, TranscribeOptions, VoiceSelect};
        kernels();
        if !synaptix_music_sheetsage2::loader::bundle_has_sheetsage2(&self.model) {
            return Err(format!("в {} нет транскриптора SheetSage2", self.model.display()));
        }
        let sheet = SheetSage2::open(&self.model, self.device, DType::BF16).map_err(|e| e.to_string())?;
        let ch = channels(audio, 48_000, 2)?;
        let samples = prepare_audio(&ch, 48_000, sheet.sample_rate()).map_err(|e| e.to_string())?;
        let options = TranscribeOptions {
            melody_only: mode != "full",
            voices: match voices {
                "vocal" => VoiceSelect::Vocal,
                "ins" => VoiceSelect::Instrumental,
                _ => VoiceSelect::Both,
            },
            ..Default::default()
        };
        let seconds = samples.len() as f64 / sheet.sample_rate() as f64;
        let window = sheet.config.input_audio_length.min(seconds).max(1.0);
        let mut on_progress = |p: Progress| match p {
            Progress::Encoding { window: w, windows } => progress((w - 1) as f32 / windows as f32),
            Progress::Decoding { window: w, windows, tokens } => {
                let part = (tokens as f64 / (12.0 * window)).min(0.95) as f32;
                progress(((w - 1) as f32 + part) / windows as f32)
            }
            Progress::Notation => progress(0.99),
        };
        let r = sheet.transcribe(&samples, &options, &mut on_progress, cancel).map_err(|e| e.to_string())?;
        r.abc.ok_or_else(|| format!("партитура не получилась: {}", r.abc_error.unwrap_or_default()))
    }
}

/// Латенты синтеза (`[кадры, 64]` или `[1, 64, кадры]`) → `[1, 64, кадры]` на хосте.
fn yue_latent(t: &Tensor, seconds: f32) -> R<YueLatent> {
    let d = t.dims().to_vec();
    let z = match d.as_slice() {
        [_, 64] => t.transpose(0, 1).and_then(|t| t.contiguous()).and_then(|t| t.unsqueeze(0)).map_err(|e| e.to_string())?,
        [1, 64, _] => t.clone(),
        other => return Err(format!("латент YuE2 формы {other:?}")),
    };
    Ok(YueLatent { latent: on_host(&z)?, seconds })
}

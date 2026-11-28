# Наш порт MiniMax-H3 (synaptix) — выдержки из кода, 2026-11-28

## config.rs (константы)
```rust
use std::path::Path;

use serde::Deserialize;

use crate::source::H3Source;
use crate::H3Error;

pub const DEFAULT_SIGMA_SHIFT_VIDEO: f32 = 12.0;
pub const DEFAULT_SIGMA_SHIFT_AUDIO: f32 = 3.0;
pub const VISUAL_COND_TIMESTEP: f32 = 0.999;
pub const AUDIO_COND_TIMESTEP: f32 = 1.0;
pub const FPS: f64 = 24.0;
pub const AUDIO_SAMPLE_RATE: usize = 32_000;
pub const AUDIO_SAMPLES_PER_LATENT: usize = 800;
pub const AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const VAE_SPATIAL_RATIO: usize = 16;
pub const VAE_TEMPORAL_RATIO: usize = 4;
pub const FRAME_GRID_STEP: usize = 17;
pub const FRAME_GRID_BASE: usize = 5;
pub const MODALITY_VIDEO: usize = 0;
pub const MODALITY_TEXT: usize = 1;
pub const MODALITY_AUDIO: usize = 2;
pub const ADALN_MODALITIES: usize = 3;
pub const ADALN_CHUNKS: usize = 6;
pub const FINAL_ADALN_CHUNKS: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H3Variant {
    Fl2va,
    Ref2va,
impl Default for H3Config {
    fn default() -> Self {
        Self {
            hidden_size: 5376,
            num_layers: 50,
            token_refiner_num_layers: 2,
            num_attention_heads: 56,
            attention_head_dim: 128,
            ffn_hidden_size: 14336,
            latents_dim: 24,
            audio_latents_dim: 32,
            patch_size: [1, 2, 2],
            text_dim: 5120,
            timestep_input_dim: 256,
            time_embed_hidden_size: 5376,
            time_embed_dim: 2688,
            adaln_out_features: 96768,
            final_adaln_out_features: 10752,
            rope_inv_freq_len: 16,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            sigma_shift_video: DEFAULT_SIGMA_SHIFT_VIDEO,
            sigma_shift_audio: DEFAULT_SIGMA_SHIFT_AUDIO,
            tasks: Vec::new(),
            partition: String::new(),
        }
    }
}
```

## spec.rs (режимы, шаги и CFG по умолчанию)
```rust
use crate::config::H3Variant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Conditioning {
    None,
    FirstFrame,
    LastFrame,
    FirstLastFrame,
    References,
    VideoAudio,
}

#[derive(Debug, Clone)]
pub struct PipelineSpec {
    pub name: &'static str,
    pub desc: &'static str,
    pub variant: H3Variant,
    pub conditioning: Conditioning,
    pub default_steps: usize,
    pub default_cfg: f32,
    pub needs_lora: bool,
    pub implemented: bool,
}

pub fn registry() -> Vec<PipelineSpec> {
    vec![
        PipelineSpec {
            name: "t2va",
            desc: "текст → видео + синхронное стерео",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::None,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "t2va-turbo",
            desc: "текст → видео + звук, Turbo LoRA (lightx2v), 8 шагов без CFG",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::None,
            default_steps: 8,
            default_cfg: 1.0,
            needs_lora: true,
            implemented: true,
        },
        PipelineSpec {
            name: "fl2va-first",
            desc: "первый кадр → видео + звук",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::FirstFrame,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "fl2va-last",
            desc: "последний кадр → видео + звук",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::LastFrame,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "fl2va-both",
            desc: "переход между первым и последним кадром",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::FirstLastFrame,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "ref2va",
            desc: "референсы (до 9 изображений, 3 видео, 3 аудио) → видео + звук",
            variant: H3Variant::Ref2va,
            conditioning: Conditioning::References,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
        PipelineSpec {
            name: "av-restyle",
            desc: "частичный денойзинг существующего видео со звуком",
            variant: H3Variant::Fl2va,
            conditioning: Conditioning::VideoAudio,
            default_steps: 20,
            default_cfg: 5.0,
            needs_lora: false,
            implemented: true,
        },
    ]
}

pub fn by_name(name: &str) -> Option<PipelineSpec> {
```

## scheduler.rs
```rust
use crate::config::{DEFAULT_SIGMA_SHIFT_AUDIO, DEFAULT_SIGMA_SHIFT_VIDEO};

pub fn shift_sigma(base: f64, shift: f64) -> f64 {
    shift * base / (1.0 + (shift - 1.0) * base)
}

pub fn unshift_sigma(sigma: f64, shift: f64) -> f64 {
    sigma / (shift + sigma * (1.0 - shift))
}

pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    shift_sigma(unshift_sigma(sigma, from_shift), to_shift)
}

#[derive(Debug, Clone)]
pub struct H3Scheduler {
    pub sigmas: Vec<f64>,
    pub shift_video: f64,
    pub shift_audio: f64,
}

impl Default for H3Scheduler {
    fn default() -> Self {
        Self::new(20, DEFAULT_SIGMA_SHIFT_VIDEO as f64, DEFAULT_SIGMA_SHIFT_AUDIO as f64)
    }
}

impl H3Scheduler {
    pub fn new(steps: usize, shift_video: f64, shift_audio: f64) -> Self {
        let (ov, oa) = crate::runtime::sigma_shift();
        let shift_video = ov.unwrap_or(shift_video);
        let shift_audio = oa.unwrap_or(shift_audio);
        let steps = steps.max(1);
        let sigmas = (0..=steps)
            .map(|i| {
                let base = 1.0 - i as f64 / steps as f64;
                shift_sigma(base, shift_video)
            })
            .collect();
        Self { sigmas, shift_video, shift_audio }
    }

    pub fn from_sigmas(sigmas: Vec<f64>, shift_video: f64, shift_audio: f64) -> Self {
        Self { sigmas, shift_video, shift_audio }
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    pub fn video_sigma(&self, step: usize) -> f64 {
        self.sigmas[step]
    }

    pub fn audio_sigma(&self, step: usize) -> f64 {
        time_shift_sigma(self.sigmas[step], self.shift_video, self.shift_audio)
    }

    pub fn audio_carry(&self, step: usize) -> f64 {
        let sv = self.sigmas[step].max(1e-6);
        self.audio_sigma(step) / sv
    }

    pub fn video_t(&self, step: usize) -> f64 {
        1.0 - self.video_sigma(step)
    }

    pub fn audio_t(&self, step: usize) -> f64 {
        1.0 - self.audio_sigma(step)
    }

    pub fn video_dt(&self, step: usize) -> f64 {
        self.sigmas[step + 1] - self.sigmas[step]
    }

    pub fn audio_dt(&self, step: usize) -> f64 {
        self.audio_sigma(step + 1) - self.audio_sigma(step)
    }
}
```

## guider.rs
```rust
use synaptix_core::{error::SynaptixError, tensor::Tensor};

type R<T> = Result<T, SynaptixError>;

#[derive(Debug, Clone, Copy)]
pub struct GuiderParams {
    pub cfg_scale: f32,
    pub rescale: f32,
    pub skip_steps: usize,
}

impl GuiderParams {
    pub fn positive_only() -> Self {
        Self { cfg_scale: 1.0, rescale: 0.0, skip_steps: 0 }
    }

    pub fn cfg(scale: f32) -> Self {
        Self { cfg_scale: scale, rescale: 0.0, skip_steps: 0 }
    }

    pub fn needs_uncond(&self, step: usize) -> bool {
        self.cfg_scale > 1.0 && step >= self.skip_steps
    }
}

pub fn apply_cfg(cond: &Tensor, uncond: &Tensor, scale: f32) -> R<Tensor> {
    if scale == 1.0 {
        return Ok(cond.clone());
    }
    let diff = cond.sub(uncond)?;
    uncond.add(&diff.mul_scalar(scale)?)
}

pub fn rescale_to(pred: &Tensor, reference: &Tensor, factor: f32) -> R<Tensor> {
    if factor <= 0.0 {
        return Ok(pred.clone());
    }
    let std_p = tensor_std(pred)?;
    let std_r = tensor_std(reference)?;
    if std_p <= 0.0 {
        return Ok(pred.clone());
    }
    let scaled = pred.mul_scalar(std_r / std_p)?;
    let a = scaled.mul_scalar(factor)?;
    let b = pred.mul_scalar(1.0 - factor)?;
    a.add(&b)
}

fn tensor_std(t: &Tensor) -> R<f32> {
    let host = t
        .to_device(synaptix_core::device::Device::Cpu)?
        .to_dtype(synaptix_core::dtype::DType::F32)?;
    let v = host.to_vec1::<f32>()?;
    if v.is_empty() {
        return Ok(0.0);
    }
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    Ok(var.sqrt())
}
```

## refs.rs (шапка)
```rust
//! Референсы партиции Ref2VA: картинки, видео (со своей дорожкой) и аудио.
//!
//! Порядок референсов — часть запроса: он задаёт номера `<Picture i>` /
//! `<Video k>` / `<Audio j>` в презентации энкодера и двигает общие
//! RoPE-часы раскладки, так что один и тот же набор в другом порядке — другой
//! запрос. Всё, что ниже, сохраняет порядок входного списка.
//!
//! Нормализация повторяет эталон (`diffusers`, `MiniMaxH3Ref2VASetupStep`):
//! картинка — на свою короткую сторону 2048 без ограничения площади, видео —
//! на 24 fps и холст своего аспекта по общему правилу 768p, дорожка — 32 кГц
//! стерео, обрезанная по длине генерации. Масштабирует ffmpeg (LANCZOS), как
//! и у эталона при декодировании.

use std::path::{Path, PathBuf};
use std::process::Command;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_vlm_qwen3::preprocess::{PreprocessLimits, VideoPixelLimits};

use crate::audio_vae::AudioVae;
use crate::config::{
    AUDIO_SAMPLE_RATE, FPS, FRAME_GRID_BASE, FRAME_GRID_STEP, VAE_SPATIAL_RATIO,
    VISUAL_COND_TIMESTEP,
};
use crate::layout::RefBlock;
use crate::pipeline::{apply_noise_aug, pack_audio, patchify_video, CondRows};
use crate::text_encoder::{EncoderHandle, ImageGrid, RefItem, VideoBlock};
use crate::vae::VaeEncoder;
use crate::H3Error;

pub use synaptix_io::refs::{has_audio_stream, labels, RefKind};

pub const MAX_IMAGES: usize = 9;
pub const MAX_VIDEOS: usize = 3;
pub const MAX_AUDIOS: usize = 3;
pub const MAX_REFS: usize = 12;

pub const CANVAS_MULTIPLE: usize = 32;
pub const CANVAS_SHORT_EDGE: usize = 768;
pub const CANVAS_MAX_PIXELS: usize = 768 * 1344;
pub const REF_IMAGE_SHORT_EDGE: usize = 2048;
/// Частота, с которой энкодер читает видео-референс.
pub const ENCODER_SAMPLE_FPS: f64 = 2.0;

/// Бюджет картинки у HF-процессора H3 (`preprocessor_config.json`): референс
/// на 2048 по короткой стороне должен дойти до башни без уменьшения.
pub const REF_IMAGE_LIMITS: PreprocessLimits =
    PreprocessLimits { min_pixels: 65_536, max_pixels: 16_777_216 };

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefSource {
    Image(PathBuf),
    /// `use_audio` — брать ли дорожку видео как его собственный `<Audio j>`.
    Video { path: PathBuf, use_audio: bool },
    Audio(PathBuf),
}

impl RefSource {
    pub fn from_path(path: impl Into<PathBuf>, use_audio: bool) -> Result<Self, H3Error> {
        let path = path.into();
```

## synaptix-gen/src/video.rs — fn h3 (служба)
```rust
182:pub fn h3(model: &Path, encoder: Option<&Path>, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
183-    use crate::h3_stages as st;
184-    use crate::tail::TailLatent;
185-    let mut m = st::open(model, encoder, device, req.quant.as_deref())?;
186-    let lora = req.lora.as_ref().filter(|_| req.lora_strength.unwrap_or(1.0) > 0.0);
187-    if let Some(p) = lora {
188-        if !p.exists() {
189-            return Err(format!("LoRA нет: {}", p.display()));
190-        }
191-        m = m.with_lora(p.clone(), req.lora_strength.unwrap_or(1.0));
192-    }
193-    // дистиллированная LoRA (Turbo): свои шаги и без CFG
194-    let distilled = lora.and_then(|p| lora_steps(p));
195-    let context = match &req.context {
196-        Some(p) => Some(TailLatent::load(p)?),
197-        None => None,
198-    };
199-    let continued = context.is_some();
200-    let ctx_lat_frames = context.as_ref().map(|c| c.meta.context_frames).unwrap_or(0);
201-    let size = st::Size { width: req.width.unwrap_or(0), height: req.height.unwrap_or(0), seconds: req.seconds.unwrap_or(0.0) };
202-    let steps = req.steps.filter(|&s| s > 0).or(distilled).unwrap_or(20);
203-    let cfg_scale = req.guidance.unwrap_or(if distilled.is_some() { 1.0 } else { 5.0 });
204-    // Негатив — только при CFG > 1 (без CFG его не кодировать).
205-    let negative = req.negative.as_deref().filter(|_| cfg_scale > 1.0);
206-    let first = if continued { None } else { req.first_frame.as_deref() };
207-    let _status = synaptix_core::gen_status::Scope::new("кодирование текста и кадров");
208-    let cond = m.encode_with_context(size, &req.prompt, negative, first, req.last_frame.as_deref(), &req.refs, st::RefSize::Match, context, req.model_id.as_deref())?;
209-    let keyframes = first.is_some() as usize + req.last_frame.is_some() as usize;
210-    synaptix_core::gen_status::phase("загрузка DiT");
211-    let lat = m.sample(&cond, steps, cfg_scale, req.seed, req.source_video.as_deref(), req.strength, cancel, progress)?;
212-    drop(cond);
213-    let tail_path = if let Some(tp) = &req.tail_out {
214-        let model_name = req.model_id.as_deref().unwrap_or("minimax-h3");
215-        let tail = lat.tail(req.context_frames.unwrap_or(0), model_name)?;
216-        tail.save(tp)?;
217-        Some(tp.clone())
218-    } else {
219-        None
220-    };
221-    synaptix_core::gen_status::phase("декодер VAE: видео");
222-    let frames = m.decode_video(&lat)?;
223-    synaptix_core::gen_status::phase("декодер VAE: звук");
224-    let (pcm, sample_rate, channels) = m.decode_audio(&lat)?;
225-    synaptix_core::gen_status::phase("запись mp4");
226-    write_mp4(&frames, Some((&pcm, sample_rate, channels)), st::FPS, out)?;
227-    let (width, height) = (lat.width, lat.height);
228-    let n_frames = frames.len();
229-    let mut label = format!("MiniMax-H3 {width}×{height}, {:.1} с, {steps} шагов", n_frames as f64 / st::FPS);
230-    if let Some(p) = lora {
231-        label += &format!(", LoRA {}", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
232-    }
233-    if continued {
234-        label += &format!(", с хвоста ({ctx_lat_frames} лат.)");
235-    } else if keyframes > 0 {
236-        label += &format!(", ключевых кадров {keyframes}");
237-    }
238-    if !req.refs.is_empty() {
239-        label += &format!(", референсов {}", req.refs.len());
240-    }
241-    Ok(VideoOutput { width, height, frames: n_frames, fps: st::FPS, label, tail: tail_path, continued_from_latent: continued })
242-}
243-
244-/// Кадры `[3,H,W]` 0..1 и звук (перемежённые отсчёты) → mp4 через ffmpeg.
245-/// Звуковые токены LTX ровно на `fa` кадров: обрезка или повтор последнего.
246-pub(crate) fn fit_audio_tokens(enc: Tensor, fa: usize) -> R<Tensor> {
247-    let far = enc.dims()[1];
248-    if far >= fa {
249-        return enc.narrow(1, 0, fa).map_err(err)?.contiguous().map_err(err);
250-    }
251-    let last = enc.narrow(1, far - 1, 1).map_err(err)?.contiguous().map_err(err)?;
252-    let mut parts = vec![enc];
117:pub fn lora_steps(p: &Path) -> Option<usize> {
118-    let n = p.file_name()?.to_string_lossy().to_ascii_lowercase();
119-    let i = n.find("step")?;
120-    let digits: String = n[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
121-    digits.parse().ok().filter(|&s: &usize| s > 0 && s <= 64)
122-}
123-
124-/// Формат весов, которым квантован бандл (`quant_manifest.json`): у
125-/// квантованного файла DiT берёт его же формат — перекодировка в другой
126-/// квант (MXFP8 → NVFP4) теряет качество, а плотное чтение деквантует
127-/// и стримит вдвое больший вес.
128-/// Чекпойнт LTX-2.5 и новее (сэмплер Euler-ancestral, подпись «LTX-2.5»);
129-/// `SYN_LTX_SAMPLER=euler` — прежний Euler.
130-pub(crate) fn is_ltx25(model: &Path) -> bool {
131-    std::env::var("SYN_LTX_SAMPLER").as_deref() != Ok("euler")
132-        && synaptix_video_ltx23::loader::LtxCheckpoint::open(model, Device::Cpu, DType::BF16)
--
1456:    fn lora_steps_from_name() {
1457-        use std::path::Path;
1458-        assert_eq!(super::lora_steps(Path::new("minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors")), Some(8));
1459-        assert_eq!(super::lora_steps(Path::new("minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors")), Some(4));
1460-        assert_eq!(super::lora_steps(Path::new("style.safetensors")), None);
1461-    }
1462-}
```

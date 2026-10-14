//! MiniMax-H3 по стадиям — для нодовых движков (граф службы synaptixd):
//! модель → размер ролика → кодировщик (описание с ключевыми кадрами или
//! референсами) → сэмплер (видео и звук вместе; исходный ролик — перерисовка)
//! → декодеры видео и звука. Веса каждая стадия грузит на время своей
//! работы и отпускает с ужатием пулов — как [`crate::video::h3`], который
//! вызывает эти же стадии подряд.

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3 as h3;

use crate::video::{media_seconds, quant_setting};

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const COMPUTE: DType = DType::BF16;

/// Модель: путь и настройки; веса — по стадиям.
pub struct H3Model {
    pub model: PathBuf,
    encoder: Option<PathBuf>,
    device: Device,
    quant: DType,
}

/// Размер и длительность ролика; 0 — умолчание (1344×768, 5 с).
#[derive(Debug, Clone, Copy, Default)]
pub struct Size {
    pub width: usize,
    pub height: usize,
    pub seconds: f64,
}

/// Условие: закодированное описание, ключевые кадры и референсы, геометрия.
pub struct H3Cond {
    source: h3::H3Source,
    geometry: h3::pipeline::Geometry,
    cond: h3::text_encoder::H3Conditioning,
    negative: Option<h3::text_encoder::H3Conditioning>,
    keyframes: Vec<(usize, Tensor)>,
    ref_media: Vec<h3::refs::RefMedia>,
    refs: usize,
}

impl H3Cond {
    pub fn label(&self) -> String {
        let g = &self.geometry;
        let mut s = format!("{}×{}, {:.1} с", g.width, g.height, g.frame_count as f64 / h3::config::FPS);
        if !self.keyframes.is_empty() {
            s += &format!(", ключевых кадров {}", self.keyframes.len());
        }
        if self.refs > 0 {
            s += &format!(", референсов {}", self.refs);
        }
        s
    }
}

/// Латенты ролика после сэмплера.
pub struct H3Latent {
    source: h3::H3Source,
    video: Tensor,
    audio: Tensor,
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub label: String,
}

/// Режим памяти и настройки рантайма H3 — перед каждой стадией (как у `synaptix h3`).
fn setup() -> R<h3::H3MemoryMode> {
    let mode = h3::H3MemoryMode::parse("auto").ok_or("режим памяти")?;
    mode.install();
    h3::runtime::set_sigma_shift(None, None);
    h3::runtime::set_vae_tile(None);
    h3::runtime::set_load_reserve_bytes(13 << 29);
    Ok(mode)
}

/// Открыть модель (`quant`: `nvfp4` по умолчанию, `mxfp8`, `bf16`).
pub fn open(model: &Path, encoder: Option<&Path>, device: Device, quant: Option<&str>) -> R<H3Model> {
    if !device.is_cuda() {
        return Err("MiniMax-H3 работает только на видеокарте".into());
    }
    if !model.exists() {
        return Err(format!("модели нет: {}", model.display()));
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    Ok(H3Model { model: model.to_path_buf(), encoder: encoder.map(Path::to_path_buf), device, quant: quant_setting(quant).unwrap_or(DType::NVFP4) })
}

fn is_audio(p: &Path) -> bool {
    matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("wav" | "mp3" | "flac" | "ogg" | "opus" | "m4a" | "aac"))
}

impl H3Model {
    pub fn is_quantized(&self) -> bool {
        self.quant.is_quantized()
    }

    /// Описание (и негатив) с ключевыми кадрами (FL2VA) или референсами
    /// (Ref2VA: картинки, видео, звук — до 12) в условие. Голос в
    /// референсах задаёт длительность ролика.
    pub fn encode(&self, size: Size, prompt: &str, negative: Option<&str>, first: Option<&Path>, last: Option<&Path>, refs: &[PathBuf]) -> R<H3Cond> {
        setup()?;
        let device = self.device;
        let snap = |v: usize| (v / 16).max(1) * 16;
        let width = snap(if size.width > 0 { size.width } else { 1344 });
        let height = snap(if size.height > 0 { size.height } else { 768 });
        let ref_sources = refs.iter().map(|p| h3::refs::RefSource::from_path(p, true)).collect::<Result<Vec<_>, _>>().map_err(err)?;
        h3::refs::validate(&ref_sources).map_err(err)?;
        if !ref_sources.is_empty() && (first.is_some() || last.is_some()) {
            return Err("референсы и ключевые кадры — разные варианты H3 (ref2va и fl2va), вместе не идут".into());
        }
        let variant = if ref_sources.is_empty() { h3::config::H3Variant::Fl2va } else { h3::config::H3Variant::Ref2va };
        let source = h3::H3Source::open(&self.model, variant).map_err(err)?;
        let audio_secs = refs.iter().filter(|p| is_audio(p)).filter_map(|p| media_seconds(p)).fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s))));
        let frame_count = h3::config::frames_for_duration(audio_secs.or((size.seconds > 0.0).then_some(size.seconds)).unwrap_or(5.0).clamp(1.0, 15.0));
        let geometry = h3::pipeline::Geometry::new(width, height, frame_count);

        let encoder_source = match &self.encoder {
            Some(p) => h3::H3EncoderSource::open(p).map_err(err)?,
            None => h3::H3EncoderSource::from_model(&source).ok_or_else(|| format!("в {} нет энкодера Qwen3-VL", source.path().display()))?,
        };
        let encoder = h3::text_encoder::EncoderHandle::load_source(&encoder_source, device, COMPUTE, self.quant).map_err(err)?;

        let mut images: Vec<(Tensor, h3::text_encoder::ImageGrid)> = Vec::new();
        let mut keyframes: Vec<(usize, Tensor)> = Vec::new();
        for (idx, p) in [(0usize, first), (geometry.frame_count - 1, last)] {
            let Some(p) = p else { continue };
            let raw = synaptix_io::image::png::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
            let img = synaptix_io::image::fit_image(&raw, geometry.width, geometry.height, true).map_err(err)?;
            images.push(encoder.prepare_image(&img).map_err(err)?);
            keyframes.push((idx, img));
        }
        let ref_media = if ref_sources.is_empty() {
            Vec::new()
        } else {
            h3::refs::decode(
                &ref_sources,
                &h3::refs::RefOptions { image_size: h3::refs::RefImageSize::Match, target_width: geometry.width, target_height: geometry.height, frame_count: geometry.frame_count },
            )
            .map_err(err)?
        };
        let merge = encoder.merge_size();
        let grids: Vec<h3::text_encoder::ImageGrid> = images.iter().map(|(_, g)| *g).collect();
        let presentation = if !ref_media.is_empty() {
            let inputs = h3::refs::encoder_inputs(&encoder, &ref_media).map_err(err)?;
            images = inputs.vision;
            h3::text_encoder::presentation_ref2va(prompt, &inputs.items, merge)
        } else if grids.is_empty() {
            h3::text_encoder::presentation_t2va(prompt)
        } else {
            h3::text_encoder::presentation_fl2va(prompt, &grids, merge)
        };
        let cond = encoder.encode(&presentation, &images).map_err(err)?;
        // Негатив работает только при CFG > 1 — это решает сэмплер.
        let negative = match negative.map(str::trim).filter(|n| !n.is_empty()) {
            Some(np) => Some(encoder.encode(&h3::text_encoder::presentation_t2va(np), &[]).map_err(err)?),
            None => None,
        };
        drop(encoder);
        h3::memory::trim_pool(device);
        Ok(H3Cond { source, geometry, cond, negative, keyframes, ref_media, refs: refs.len() })
    }

    /// Денойз видео и звука. `source_video` — перерисовка ролика с силой
    /// `strength` (0..1, по умолчанию 0.6). `progress(шаг, всего)`.
    #[allow(clippy::too_many_arguments)]
    pub fn sample(&self, c: &H3Cond, steps: usize, cfg_scale: f32, seed: u64, source_video: Option<&Path>, strength: Option<f32>, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<H3Latent> {
        let mode = setup()?;
        let device = self.device;
        let quant_dit = self.quant;
        // Квантованный DiT — и внимание в FP8 (e4m3 QKᵀ и P·V).
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(quant_dit.is_quantized());
        let steps = if steps > 0 { steps } else { 20 };
        let geometry = c.geometry;
        let source = &c.source;
        let ckpt = h3::H3Checkpoint::open_source(source.clone(), device, COMPUTE).map_err(err)?;
        let mut sched = h3::H3Scheduler::new(steps, ckpt.config.sigma_shift_video as f64, ckpt.config.sigma_shift_audio as f64);
        let conditioning = h3::pipeline::Conditioning { context: c.cond.hidden.clone(), text_tags: c.cond.tags.clone() };
        let mut dreq = h3::pipeline::DenoiseRequest::new(geometry, &conditioning);
        let neg_cond;
        if let (Some(n), true) = (&c.negative, cfg_scale > 1.0) {
            neg_cond = h3::pipeline::Conditioning { context: n.hidden.clone(), text_tags: n.tags.clone() };
            dreq.negative = Some(&neg_cond);
            dreq.guider = h3::guider::GuiderParams { cfg_scale, rescale: 0.0, skip_steps: 0 };
        }
        dreq.seed = Some(seed);
        dreq.keyframes = c.keyframes.iter().map(|(i, _)| h3::layout::Keyframe { resolved_frame_index: *i }).collect();

        let ref_media = &c.ref_media;
        if !ref_media.is_empty() {
            if !ckpt.config.supports_references() {
                return Err(format!("{} — вариант FL2VA, референсы понимает только Ref2VA", source.path().display()));
            }
            let vw = h3::loader::ComponentLoader::open_component(source, h3::H3Component::VideoVae, device).map_err(err)?;
            let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
            let has_audio = ref_media.iter().any(|m| match m {
                h3::refs::RefMedia::Audio(_) => true,
                h3::refs::RefMedia::Video(v) => v.audio.is_some(),
                h3::refs::RefMedia::Image(_) => false,
            });
            let audio_enc = if has_audio {
                let aw = h3::loader::ComponentLoader::open_component(source, h3::H3Component::AudioVae, device).map_err(err)?;
                Some(h3::audio_vae::AudioVae::load_full(&aw, ckpt.audio_vae_config().map_err(err)?, device, COMPUTE).map_err(err)?)
            } else {
                None
            };
            let latents = h3::refs::encode_latents(ref_media, &enc, audio_enc.as_ref(), ckpt.config.patch_size, device, seed).map_err(err)?;
            drop(enc);
            drop(audio_enc);
            h3::memory::trim_pool(device);
            dreq.refs = latents.blocks;
            dreq.cond_rows = latents.cond_rows;
        }
        // Restyle: исходный ролик (видео + звук) → латенты, денойз с середины расписания.
        if let Some(src_path) = source_video {
            if !ref_media.is_empty() || !c.keyframes.is_empty() {
                return Err("перерисовка ролика не сочетается с референсами и ключевыми кадрами".into());
            }
            let strength = strength.unwrap_or(0.6).clamp(0.05, 1.0);
            let vw = h3::loader::ComponentLoader::open_component(source, h3::H3Component::VideoVae, device).map_err(err)?;
            let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
            let aw = h3::loader::ComponentLoader::open_component(source, h3::H3Component::AudioVae, device).map_err(err)?;
            let aenc = h3::audio_vae::AudioVae::load_full(&aw, ckpt.audio_vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
            let src = h3::restyle::encode_source(src_path, geometry, &enc, &aenc, device).map_err(err)?;
            drop(enc);
            drop(aenc);
            h3::memory::trim_pool(device);
            let (tail, v0, a0) = h3::restyle::start(&sched, strength, &src, seed, device, COMPUTE).map_err(err)?;
            sched = tail;
            dreq.init_video = Some(v0);
            dreq.init_audio = Some(a0);
        }
        let dit = h3::dit::H3Dit::load(&ckpt, device, COMPUTE, quant_dit).map_err(err)?;
        let prep = h3::pipeline::prepare(&dit, &dreq, &sched).map_err(err)?;
        if !c.keyframes.is_empty() {
            let vw = h3::loader::ComponentLoader::open_component(source, h3::H3Component::VideoVae, device).map_err(err)?;
            let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
            let mut latents = Vec::with_capacity(c.keyframes.len());
            for (_, img) in &c.keyframes {
                let d = img.dims().to_vec();
                let x = img.to_device(device).map_err(err)?.reshape(vec![1, d[0], 1, d[1], d[2]]).map_err(err)?.mul_scalar(2.0).map_err(err)?.add_scalar(-1.0).map_err(err)?;
                latents.push(enc.encode(&x).map_err(err)?);
            }
            drop(enc);
            h3::memory::trim_pool(device);
            dreq.cond_rows.video = h3::pipeline::cond_rows_from_keyframe_latents(&latents, dit.cfg.patch_size, None, seed).map_err(err)?;
        }
        let _plan = h3::memory::plan(&ckpt, &prep.plan, prep.layout.seq_len, quant_dit, COMPUTE, COMPUTE, mode, device).map_err(err)?;
        let cache = h3::pipeline::build_adaln_cache(&dit, &ckpt, &prep, COMPUTE).map_err(err)?;
        let on_step = |p: h3::pipeline::DenoiseProgress| progress(p.step, p.total);
        let hooks = h3::pipeline::DenoiseHooks { progress: Some(&on_step), cancel: Some(cancel) };
        let lat = h3::pipeline::denoise_av(&dit, &cache, &prep, &dreq, &sched, &hooks).map_err(err)?;
        drop(cache);
        drop(prep);
        drop(dit);
        h3::memory::trim_pool(device);
        let label = format!("MiniMax-H3 {}, {steps} шагов", c.label());
        Ok(H3Latent { source: source.clone(), video: lat.video_latent, audio: lat.audio_latent, width: geometry.width, height: geometry.height, frames: geometry.frame_count, label })
    }

    /// Латент видео → кадры `[3,H,W]` 0..1 (24 к/с).
    pub fn decode_video(&self, lat: &H3Latent) -> R<Vec<Tensor>> {
        setup()?;
        let device = self.device;
        let ckpt = h3::H3Checkpoint::open_source(lat.source.clone(), device, COMPUTE).map_err(err)?;
        let vw = h3::loader::ComponentLoader::open_component(&lat.source, h3::H3Component::VideoVae, device).map_err(err)?;
        let vdec = h3::vae::VaeDecoder::load(&vw, ckpt.vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
        let rgb = vdec.decode(&lat.video).map_err(err)?;
        drop(vdec);
        drop(vw);
        h3::memory::trim_pool(device);
        h3::vae::rgb_to_frames(&rgb).map_err(err)
    }

    /// Латент звука → перемежённые отсчёты, частота, каналы.
    pub fn decode_audio(&self, lat: &H3Latent) -> R<(Vec<f32>, u32, u16)> {
        setup()?;
        let device = self.device;
        let ckpt = h3::H3Checkpoint::open_source(lat.source.clone(), device, COMPUTE).map_err(err)?;
        let aw = h3::loader::ComponentLoader::open_component(&lat.source, h3::H3Component::AudioVae, device).map_err(err)?;
        let adec = h3::audio_vae::AudioVae::load_decoder(&aw, ckpt.audio_vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
        let wave = adec.decode(&lat.audio).map_err(err)?;
        let sample_rate = adec.sample_rate();
        drop(adec);
        h3::memory::trim_pool(device);
        let pcm = h3::audio_vae::interleave_stereo(&wave).map_err(err)?;
        let channels = wave.dims()[1] as u16;
        Ok((pcm, sample_rate as u32, channels))
    }
}

/// Кадров в секунду у H3.
pub const FPS: f64 = h3::config::FPS;

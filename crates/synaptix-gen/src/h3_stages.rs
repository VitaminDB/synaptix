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

use crate::tail::{TailLatent, TailMeta};
use crate::video::{media_seconds, quant_setting};

/// Число латентных кадров хвоста по умолчанию для H3 (7 лат. кадров = 22 пикс. ≈ 0.92 с,
/// кратно 5-токенной фазе ViT VAE: 5n + 2).
pub const DEFAULT_CONTEXT_FRAMES: usize = 7;

/// Подогнать запрошенное число латентных кадров хвоста к фазово-выровненной
/// сетке H3 (`5m + 2`: 2 → 5 пикс. к., 7 → 22 пикс. к., 12 → 39 пикс. к.)
/// и гарантировать запас хотя бы в один новый блок до `max_latent_t`.
pub fn snap_context_frames(requested: usize, max_latent_t: usize) -> usize {
    let want = if requested == 0 { DEFAULT_CONTEXT_FRAMES } else { requested };
    let mut snapped = ((want + 2) / 5) * 5 + 2;
    let limit = max_latent_t.saturating_sub(5).max(2);
    while snapped > limit && snapped > 2 {
        snapped -= 5;
    }
    snapped.min(max_latent_t.saturating_sub(1).max(1))
}

/// Число пиксельных кадров, соответствующее `t_ctx` латентным кадрам хвоста H3 (`5m + 2` → `17m + 5`).
pub fn overlap_frames_for_context(t_ctx: usize) -> usize {
    if t_ctx <= 2 {
        h3::config::FRAME_GRID_BASE
    } else {
        (t_ctx.saturating_sub(2) / 5) * h3::config::FRAME_GRID_STEP + h3::config::FRAME_GRID_BASE
    }
}

/// Число аудио-латентных кадров (40 Гц), соответствующее `t_ctx` видеолатентам хвоста H3.
pub fn audio_frames_for_context(t_ctx: usize) -> usize {
    let px = overlap_frames_for_context(t_ctx);
    ((px as f64 / FPS) * h3::config::AUDIO_LATENTS_PER_SECOND as f64).round() as usize
}

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
    /// LoRA поверх DiT (Turbo и т. п.) и её сила.
    lora: Option<(PathBuf, f32)>,
}

/// Размер и длительность ролика; 0 — умолчание (1344×768, 5 с).
#[derive(Debug, Clone, Copy, Default)]
pub struct Size {
    pub width: usize,
    pub height: usize,
    pub seconds: f64,
}

/// Картинки-референсы: под площадь кадра или крупно (2048 по короткой стороне).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum RefSize {
    #[default]
    Match,
    Max,
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
    context: Option<TailLatent>,
}

impl H3Cond {
    pub fn label(&self) -> String {
        let g = &self.geometry;
        let out_frames = g.frame_count.saturating_sub(self.context.as_ref().map(|t| t.meta.overlap_frames).unwrap_or(0));
        let mut s = format!("{}×{}, {:.1} с", g.width, g.height, out_frames as f64 / h3::config::FPS);
        if let Some(ctx) = &self.context {
            s += &format!(", с хвоста ({} лат.)", ctx.meta.context_frames);
        } else if !self.keyframes.is_empty() {
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
    /// Число выходных кадров (уже без перекрытия хвоста).
    pub frames: usize,
    /// Сколько первых пиксельных кадров — повтор хвоста предыдущего ролика (выбрасываются при декоде).
    pub overlap_frames: usize,
    pub label: String,
}

impl H3Latent {
    /// Извлечь латент хвоста (видео + звук) прямо из выхода денойзера до декодирования VAE.
    pub fn tail(&self, context_frames: usize, model: &str) -> R<TailLatent> {
        let vt_total = self.video.dims()[2];
        let at_total = self.audio.dims()[3];
        let t_ctx = snap_context_frames(context_frames, vt_total);
        let overlap_frames = overlap_frames_for_context(t_ctx);
        let a_ctx = audio_frames_for_context(t_ctx).min(at_total.saturating_sub(1).max(1));
        let v_tail = self
            .video
            .narrow(2, vt_total - t_ctx, t_ctx)
            .map_err(err)?
            .contiguous()
            .map_err(err)?
            .to_device(Device::Cpu)
            .map_err(err)?
            .to_dtype(DType::F32)
            .map_err(err)?;
        let a_tail = self
            .audio
            .narrow(3, at_total - a_ctx, a_ctx)
            .map_err(err)?
            .contiguous()
            .map_err(err)?
            .to_device(Device::Cpu)
            .map_err(err)?
            .to_dtype(DType::F32)
            .map_err(err)?;
        Ok(TailLatent {
            meta: TailMeta {
                family: "h3".into(),
                model: model.to_string(),
                width: self.width,
                height: self.height,
                fps: FPS,
                context_frames: t_ctx,
                overlap_frames,
                audio_frames: a_ctx,
            },
            video: v_tail,
            audio: Some(a_tail),
            video_stage1: None,
        })
    }
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
    Ok(H3Model { model: model.to_path_buf(), encoder: encoder.map(Path::to_path_buf), device, quant: quant_setting(quant).unwrap_or(DType::NVFP4), lora: None })
}

impl H3Model {
    /// LoRA поверх DiT (сила > 0).
    pub fn with_lora(mut self, path: PathBuf, strength: f32) -> Self {
        self.lora = (strength > 0.0).then_some((path, strength));
        self
    }
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
    #[allow(clippy::too_many_arguments)]
    pub fn encode(&self, size: Size, prompt: &str, negative: Option<&str>, first: Option<&Path>, last: Option<&Path>, refs: &[PathBuf], ref_size: RefSize) -> R<H3Cond> {
        self.encode_with_context(size, prompt, negative, first, last, refs, ref_size, None, None)
    }

    /// То же с опциональным латентом хвоста `context` (продолжение ролика без PNG).
    /// Если подан `context`: `first` игнорируется, размер берётся из хвоста (несовпадение
    /// модели/размера — явная ошибка), а длина `size.seconds` считается без перекрытия.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_with_context(
        &self,
        size: Size,
        prompt: &str,
        negative: Option<&str>,
        first: Option<&Path>,
        last: Option<&Path>,
        refs: &[PathBuf],
        ref_size: RefSize,
        context: Option<TailLatent>,
        model_id: Option<&str>,
    ) -> R<H3Cond> {
        setup()?;
        let device = self.device;
        // Кратно 32: VAE ×16 и патч DiT 2×2 (720 → 704; при 16 падало на reshape).
        let snap = |v: usize| (v / 32).max(1) * 32;
        let (width, height, overlap_frames) = if let Some(ref ctx) = context {
            let req_w = (size.width > 0).then(|| snap(size.width));
            let req_h = (size.height > 0).then(|| snap(size.height));
            ctx.meta.check_compat("h3", model_id, req_w, req_h, Some(FPS))?;
            (ctx.meta.width, ctx.meta.height, ctx.meta.overlap_frames)
        } else {
            (
                snap(if size.width > 0 { size.width } else { 1344 }),
                snap(if size.height > 0 { size.height } else { 768 }),
                0,
            )
        };
        let first = if context.is_some() { None } else { first };
        let ref_sources = refs.iter().map(|p| h3::refs::RefSource::from_path(p, true)).collect::<Result<Vec<_>, _>>().map_err(err)?;
        h3::refs::validate(&ref_sources).map_err(err)?;
        if !ref_sources.is_empty() && (first.is_some() || last.is_some()) {
            return Err("референсы и ключевые кадры — разные варианты H3 (ref2va и fl2va), вместе не идут".into());
        }
        let variant = if ref_sources.is_empty() { h3::config::H3Variant::Fl2va } else { h3::config::H3Variant::Ref2va };
        let source = h3::H3Source::open(&self.model, variant).map_err(err)?;
        let audio_secs = refs.iter().filter(|p| is_audio(p)).filter_map(|p| media_seconds(p)).fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s))));
        let want_secs = audio_secs.or((size.seconds > 0.0).then_some(size.seconds)).unwrap_or(5.0).clamp(1.0, 15.0);
        let frame_count = if overlap_frames > 0 {
            let want_new = (want_secs * FPS).round() as usize;
            h3::config::snap_frame_count(overlap_frames + want_new)
        } else {
            h3::config::frames_for_duration(want_secs)
        };
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
                &h3::refs::RefOptions {
                    image_size: match ref_size {
                        RefSize::Match => h3::refs::RefImageSize::Match,
                        RefSize::Max => h3::refs::RefImageSize::Max,
                    },
                    target_width: geometry.width, target_height: geometry.height, frame_count: geometry.frame_count },
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
        Ok(H3Cond { source, geometry, cond, negative, keyframes, ref_media, refs: refs.len(), context })
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
        let mut ckpt = h3::H3Checkpoint::open_source(source.clone(), device, COMPUTE).map_err(err)?;
        if let Some((p, s)) = &self.lora {
            let lw = h3::LoraWeights::open(p, device, *s).map_err(|e| format!("LoRA {}: {e}", p.display()))?;
            ckpt = ckpt.with_lora(std::sync::Arc::new(lw));
        }
        let mut sched = h3::H3Scheduler::new(steps, ckpt.config.sigma_shift_video as f64, ckpt.config.sigma_shift_audio as f64);
        let conditioning = h3::pipeline::Conditioning { context: c.cond.hidden.clone(), text_tags: c.cond.tags.clone() };
        let mut dreq = h3::pipeline::DenoiseRequest::new(geometry, &conditioning);
        if let Some((p, _)) = &self.lora {
            let fname = p.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            if fname.contains("turbo") && std::env::var("H3_SAMPLER").is_err() {
                dreq.sampler = h3::pipeline::SamplerKind::Euler;
                eprintln!("[h3] Turbo LoRA: сэмплер Euler");
            }
        }
        let neg_cond;
        if let (Some(n), true) = (&c.negative, cfg_scale > 1.0) {
            neg_cond = h3::pipeline::Conditioning { context: n.hidden.clone(), text_tags: n.tags.clone() };
            dreq.negative = Some(&neg_cond);
            dreq.guider = h3::guider::GuiderParams { cfg_scale, rescale: 0.0, skip_steps: 0 };
        }
        dreq.seed = Some(seed);
        dreq.keyframes = c.keyframes.iter().map(|(i, _)| h3::layout::Keyframe { resolved_frame_index: *i }).collect();
        let overlap_frames = if let Some(ref ctx) = c.context {
            dreq.prefix_video = Some(ctx.video.clone());
            dreq.prefix_audio = ctx.audio.clone();
            ctx.meta.overlap_frames
        } else {
            0
        };

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
        // Ключевые кадры кодируем через VAE до загрузки DiT (как refs и restyle),
        // чтобы не держать лишний запас VRAM поверх загруженного DiT.
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
            dreq.cond_rows.video = h3::pipeline::cond_rows_from_keyframe_latents(&latents, ckpt.config.patch_size, None, seed).map_err(err)?;
        }
        h3::runtime::set_load_reserve_bytes(h3::memory::load_reserve_for(&geometry, &ckpt.config, COMPUTE, !c.keyframes.is_empty() || c.refs > 0 || c.context.is_some(), self.lora.is_some()));
        let dit = h3::dit::H3Dit::load(&ckpt, device, COMPUTE, quant_dit).map_err(err)?;
        let prep = h3::pipeline::prepare(&dit, &dreq, &sched).map_err(err)?;
        synaptix_core::gen_status::phase("денойз");
        synaptix_core::gen_status::blocks(dit.resident_blocks(), ckpt.config.num_layers);
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
        let out_frames = geometry.frame_count.saturating_sub(overlap_frames);
        Ok(H3Latent { source: source.clone(), video: lat.video_latent, audio: lat.audio_latent, width: geometry.width, height: geometry.height, frames: out_frames, overlap_frames, label })
    }

    /// Латент видео → кадры `[3,H,W]` 0..1 (24 к/с). При продолжении с хвоста
    /// декодируется полный латент (для прогрева ViT VAE), затем перекрытие отрезается.
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
        let mut frames = h3::vae::rgb_to_frames(&rgb).map_err(err)?;
        if lat.overlap_frames > 0 && frames.len() > lat.overlap_frames {
            frames.drain(0..lat.overlap_frames);
        }
        Ok(frames)
    }

    /// Латент звука → перемежённые отсчёты, частота, каналы (с обрезкой перекрытия хвоста).
    pub fn decode_audio(&self, lat: &H3Latent) -> R<(Vec<f32>, u32, u16)> {
        setup()?;
        let device = self.device;
        let ckpt = h3::H3Checkpoint::open_source(lat.source.clone(), device, COMPUTE).map_err(err)?;
        let aw = h3::loader::ComponentLoader::open_component(&lat.source, h3::H3Component::AudioVae, device).map_err(err)?;
        let adec = h3::audio_vae::AudioVae::load_decoder(&aw, ckpt.audio_vae_config().map_err(err)?, device, COMPUTE).map_err(err)?;
        let wave = adec.decode(&lat.audio).map_err(err)?;
        let sample_rate = adec.sample_rate() as u32;
        drop(adec);
        h3::memory::trim_pool(device);
        let mut pcm = h3::audio_vae::interleave_stereo(&wave).map_err(err)?;
        let channels = wave.dims()[1] as u16;
        if lat.overlap_frames > 0 {
            let cut_frames = ((lat.overlap_frames as f64 / FPS) * sample_rate as f64).round() as usize;
            let cut_samples = (cut_frames * channels as usize).min(pcm.len());
            if cut_samples > 0 && cut_samples < pcm.len() {
                pcm.drain(0..cut_samples);
            }
        }
        Ok((pcm, sample_rate, channels))
    }
}

/// Кадров в секунду у H3.
pub const FPS: f64 = h3::config::FPS;

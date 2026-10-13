//! Видео одним вызовом. MiniMax-H3: видео со звуком по описанию, по первому
//! и/или последнему кадру (fl2va). Порядок загрузки этапов — как у
//! `synaptix h3`: энкодер → (VAE ключевых кадров) → DiT → VAE видео → VAE
//! звука, между этапами пулы ужимаются. Прогресс по шагам денойза, отмена
//! флагом. Результат — mp4 (ffmpeg).

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::AtomicBool;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_video_minimax_h3 as h3;

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

#[derive(Debug, Clone, Default)]
pub struct VideoRequest {
    pub prompt: String,
    pub negative: Option<String>,
    pub width: Option<usize>,
    pub height: Option<usize>,
    /// Длительность, секунды.
    pub seconds: Option<f64>,
    pub steps: Option<usize>,
    pub guidance: Option<f32>,
    pub seed: u64,
    /// Первый кадр (картинка).
    pub first_frame: Option<PathBuf>,
    /// Последний кадр (картинка).
    pub last_frame: Option<PathBuf>,
    /// Референсы (ref2va): картинки (до 9), видео (до 3), звук (до 3 — голос
    /// для липсинка, музыка), всего до 12. Не сочетаются с ключевыми кадрами.
    pub refs: Vec<PathBuf>,
    /// Квант трансформера и энкодера: `nvfp4`, `mxfp8`, `bf16` (плотный);
    /// пусто/`auto` — умолчание модели.
    pub quant: Option<String>,
    /// LTX: звук, под который строится видео (липсинк речи, движение под
    /// музыку); с `control_video` — LipDub (переозвучка ролика).
    pub audio: Option<PathBuf>,
    /// LTX: исходное видео для retake (`retake` — отрезок, с) или продления (`extend`).
    pub source_video: Option<PathBuf>,
    pub retake: Option<(f64, f64)>,
    pub extend: bool,
    /// LTX IC-LoRA: видео-образец движения/структуры; `control` — `canny`
    /// (контуры, union-control) или `motion` (motion-track).
    pub control_video: Option<PathBuf>,
    pub control: Option<String>,
    /// LTX: две стадии (половинное разрешение → апскейлер ×2 → доводка).
    pub two_stage: bool,
    /// H3 restyle: сила перерисовки `source_video` (0..1; по умолчанию 0.6).
    pub strength: Option<f32>,
}

/// Квант из настройки; `None` — умолчание пайплайна.
fn quant_setting(q: Option<&str>) -> Option<DType> {
    match q.unwrap_or("").to_ascii_lowercase().as_str() {
        "nvfp4" => Some(DType::NVFP4),
        "mxfp8" | "fp8" => Some(DType::MXFP8),
        "bf16" | "dense" | "none" => Some(DType::BF16),
        _ => None,
    }
}

pub struct VideoOutput {
    pub width: usize,
    pub height: usize,
    pub frames: usize,
    pub fps: f64,
    pub label: String,
}

/// H3: видео в `out` (mp4). `progress(шаг, всего)`.
pub fn h3(model: &Path, encoder: Option<&Path>, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
    if !device.is_cuda() {
        return Err("MiniMax-H3 работает только на видеокарте".into());
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mode = h3::H3MemoryMode::parse("auto").ok_or("режим памяти")?;
    mode.install();
    let compute = DType::BF16;
    let q = quant_setting(req.quant.as_deref()).unwrap_or(DType::NVFP4);
    let (quant_dit, quant_enc) = (q, q);
    // Квантованный DiT — и внимание в FP8 (e4m3 QKᵀ и P·V).
    let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(quant_dit.is_quantized());
    h3::runtime::set_sigma_shift(None, None);
    h3::runtime::set_vae_tile(None);
    h3::runtime::set_load_reserve_bytes(13 << 29);
    let snap = |v: usize| (v / 16).max(1) * 16;
    let width = snap(req.width.filter(|&v| v > 0).unwrap_or(1344));
    let height = snap(req.height.filter(|&v| v > 0).unwrap_or(768));
    let steps = req.steps.filter(|&s| s > 0).unwrap_or(20);
    let cfg_scale = req.guidance.unwrap_or(5.0);
    let seed = req.seed;

    let ref_sources = req.refs.iter().map(|p| h3::refs::RefSource::from_path(p, true)).collect::<Result<Vec<_>, _>>().map_err(err)?;
    h3::refs::validate(&ref_sources).map_err(err)?;
    if !ref_sources.is_empty() && (req.first_frame.is_some() || req.last_frame.is_some()) {
        return Err("референсы и ключевые кадры — разные варианты H3 (ref2va и fl2va), вместе не идут".into());
    }
    let variant = if ref_sources.is_empty() { h3::config::H3Variant::Fl2va } else { h3::config::H3Variant::Ref2va };
    let source = h3::H3Source::open(model, variant).map_err(err)?;
    // Звук в референсах (голос для липсинка) задаёт длину ролика.
    let audio_secs = req
        .refs
        .iter()
        .filter(|p| matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("wav" | "mp3" | "flac" | "ogg" | "opus" | "m4a" | "aac")))
        .filter_map(|p| media_seconds(p))
        .fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| m.max(s))));
    let frame_count = h3::config::frames_for_duration(audio_secs.or(req.seconds.filter(|s| *s > 0.0)).unwrap_or(5.0).clamp(1.0, 15.0));
    let geometry = h3::pipeline::Geometry::new(width, height, frame_count);

    let encoder_source = match encoder {
        Some(p) => h3::H3EncoderSource::open(p).map_err(err)?,
        None => h3::H3EncoderSource::from_model(&source).ok_or_else(|| format!("в {} нет энкодера Qwen3-VL", source.path().display()))?,
    };
    let encoder = h3::text_encoder::EncoderHandle::load_source(&encoder_source, device, compute, quant_enc).map_err(err)?;

    let mut images: Vec<(Tensor, h3::text_encoder::ImageGrid)> = Vec::new();
    let mut keyframes: Vec<(usize, Tensor)> = Vec::new();
    for (idx, p) in [(0usize, &req.first_frame), (geometry.frame_count - 1, &req.last_frame)] {
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
        h3::text_encoder::presentation_ref2va(&req.prompt, &inputs.items, merge)
    } else if grids.is_empty() {
        h3::text_encoder::presentation_t2va(&req.prompt)
    } else {
        h3::text_encoder::presentation_fl2va(&req.prompt, &grids, merge)
    };
    let cond = encoder.encode(&presentation, &images).map_err(err)?;
    let negative = match req.negative.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        Some(np) if cfg_scale > 1.0 => Some(encoder.encode(&h3::text_encoder::presentation_t2va(np), &[]).map_err(err)?),
        _ => None,
    };
    drop(encoder);
    h3::memory::trim_pool(device);

    let ckpt = h3::H3Checkpoint::open_source(source.clone(), device, compute).map_err(err)?;
    let mut sched = h3::H3Scheduler::new(steps, ckpt.config.sigma_shift_video as f64, ckpt.config.sigma_shift_audio as f64);
    let conditioning = h3::pipeline::Conditioning { context: cond.hidden, text_tags: cond.tags };
    let mut dreq = h3::pipeline::DenoiseRequest::new(geometry, &conditioning);
    let neg_cond;
    if let Some(n) = negative {
        neg_cond = h3::pipeline::Conditioning { context: n.hidden, text_tags: n.tags };
        dreq.negative = Some(&neg_cond);
        dreq.guider = h3::guider::GuiderParams { cfg_scale, rescale: 0.0, skip_steps: 0 };
    }
    dreq.seed = Some(seed);
    dreq.keyframes = keyframes.iter().map(|(i, _)| h3::layout::Keyframe { resolved_frame_index: *i }).collect();

    if !ref_media.is_empty() {
        if !ckpt.config.supports_references() {
            return Err(format!("{} — вариант FL2VA, референсы понимает только Ref2VA", source.path().display()));
        }
        let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
        let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
        let has_audio = ref_media.iter().any(|m| match m {
            h3::refs::RefMedia::Audio(_) => true,
            h3::refs::RefMedia::Video(v) => v.audio.is_some(),
            h3::refs::RefMedia::Image(_) => false,
        });
        let audio_enc = if has_audio {
            let aw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::AudioVae, device).map_err(err)?;
            Some(h3::audio_vae::AudioVae::load_full(&aw, ckpt.audio_vae_config().map_err(err)?, device, compute).map_err(err)?)
        } else {
            None
        };
        let latents = h3::refs::encode_latents(&ref_media, &enc, audio_enc.as_ref(), ckpt.config.patch_size, device, seed).map_err(err)?;
        drop(enc);
        drop(audio_enc);
        h3::memory::trim_pool(device);
        dreq.refs = latents.blocks;
        dreq.cond_rows = latents.cond_rows;
    }
    let ref_media_empty = ref_media.is_empty();
    drop(ref_media);
    // Restyle: исходный ролик (видео + звук) → латенты, денойз с середины расписания.
    if let Some(src_path) = &req.source_video {
        if !ref_media_empty || !keyframes.is_empty() {
            return Err("перерисовка ролика не сочетается с референсами и ключевыми кадрами".into());
        }
        let strength = req.strength.unwrap_or(0.6).clamp(0.05, 1.0);
        let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
        let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
        let aw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::AudioVae, device).map_err(err)?;
        let aenc = h3::audio_vae::AudioVae::load_full(&aw, ckpt.audio_vae_config().map_err(err)?, device, compute).map_err(err)?;
        let src = h3::restyle::encode_source(src_path, geometry, &enc, &aenc, device).map_err(err)?;
        drop(enc);
        drop(aenc);
        h3::memory::trim_pool(device);
        let (tail, v0, a0) = h3::restyle::start(&sched, strength, &src, seed, device, compute).map_err(err)?;
        sched = tail;
        dreq.init_video = Some(v0);
        dreq.init_audio = Some(a0);
    }
    let dit = h3::dit::H3Dit::load(&ckpt, device, compute, quant_dit).map_err(err)?;
    let prep = h3::pipeline::prepare(&dit, &dreq, &sched).map_err(err)?;
    if !keyframes.is_empty() {
        let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
        let enc = h3::vae::VaeEncoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
        let mut latents = Vec::with_capacity(keyframes.len());
        for (_, img) in &keyframes {
            let d = img.dims().to_vec();
            let x = img.to_device(device).map_err(err)?.reshape(vec![1, d[0], 1, d[1], d[2]]).map_err(err)?.mul_scalar(2.0).map_err(err)?.add_scalar(-1.0).map_err(err)?;
            latents.push(enc.encode(&x).map_err(err)?);
        }
        drop(enc);
        h3::memory::trim_pool(device);
        dreq.cond_rows.video = h3::pipeline::cond_rows_from_keyframe_latents(&latents, dit.cfg.patch_size, None, seed).map_err(err)?;
    }
    let _plan = h3::memory::plan(&ckpt, &prep.plan, prep.layout.seq_len, quant_dit, compute, compute, mode, device).map_err(err)?;
    let cache = h3::pipeline::build_adaln_cache(&dit, &ckpt, &prep, compute).map_err(err)?;
    let on_step = |p: h3::pipeline::DenoiseProgress| progress(p.step, p.total);
    let hooks = h3::pipeline::DenoiseHooks { progress: Some(&on_step), cancel: Some(cancel) };
    let lat = h3::pipeline::denoise_av(&dit, &cache, &prep, &dreq, &sched, &hooks).map_err(err)?;
    drop(cache);
    drop(prep);
    drop(dit);
    h3::memory::trim_pool(device);

    let vw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::VideoVae, device).map_err(err)?;
    let vdec = h3::vae::VaeDecoder::load(&vw, ckpt.vae_config().map_err(err)?, device, compute).map_err(err)?;
    let rgb = vdec.decode(&lat.video_latent).map_err(err)?;
    drop(vdec);
    drop(vw);
    h3::memory::trim_pool(device);
    let aw = h3::loader::ComponentLoader::open_component(&source, h3::H3Component::AudioVae, device).map_err(err)?;
    let adec = h3::audio_vae::AudioVae::load_decoder(&aw, ckpt.audio_vae_config().map_err(err)?, device, compute).map_err(err)?;
    let wave = adec.decode(&lat.audio_latent).map_err(err)?;
    let sample_rate = adec.sample_rate();
    drop(adec);
    h3::memory::trim_pool(device);

    let frames = h3::vae::rgb_to_frames(&rgb).map_err(err)?;
    let pcm = h3::audio_vae::interleave_stereo(&wave).map_err(err)?;
    let channels = wave.dims()[1] as u16;
    write_mp4(&frames, Some((&pcm, sample_rate as u32, channels)), h3::config::FPS, out)?;
    let label = format!("MiniMax-H3 {width}×{height}, {:.1} с, {steps} шагов{}", geometry.frame_count as f64 / h3::config::FPS, if keyframes.is_empty() { String::new() } else { format!(", ключевых кадров {}", keyframes.len()) }) + &if req.refs.is_empty() { String::new() } else { format!(", референсов {}", req.refs.len()) };
    Ok(VideoOutput { width, height, frames: geometry.frame_count, fps: h3::config::FPS, label })
}

/// Кадры `[3,H,W]` 0..1 и звук (перемежённые отсчёты) → mp4 через ffmpeg.
/// Звуковые токены LTX ровно на `fa` кадров: обрезка или повтор последнего.
fn fit_audio_tokens(enc: Tensor, fa: usize) -> R<Tensor> {
    let far = enc.dims()[1];
    if far >= fa {
        return enc.narrow(1, 0, fa).map_err(err)?.contiguous().map_err(err);
    }
    let last = enc.narrow(1, far - 1, 1).map_err(err)?.contiguous().map_err(err)?;
    let mut parts = vec![enc];
    parts.extend((far..fa).map(|_| last.clone()));
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 1).map_err(err)?.contiguous().map_err(err)
}

/// Есть ли в файле звуковая дорожка (ffprobe).
fn has_audio_stream(p: &Path) -> bool {
    std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0"])
        .arg(p)
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false)
}

pub fn write_mp4(frames: &[Tensor], audio: Option<(&[f32], u32, u16)>, fps: f64, out: &Path) -> R<()> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("synaptix_gen_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(err)?;
    let result = (|| -> R<()> {
        for (i, fr) in frames.iter().enumerate() {
            let (h, w) = (fr.dims()[1], fr.dims()[2]);
            let planar: Vec<f32> = fr.to_device(Device::Cpu).map_err(err)?.to_dtype(DType::F32).map_err(err)?.reshape(vec![3 * h * w]).map_err(err)?.to_vec1::<f32>().map_err(err)?;
            let mut ppm = format!("P6\n{w} {h}\n255\n").into_bytes();
            ppm.reserve(3 * h * w);
            for y in 0..h {
                for x in 0..w {
                    for c in 0..3 {
                        ppm.push((planar[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0).round() as u8);
                    }
                }
            }
            std::fs::write(dir.join(format!("f{i:05}.ppm")), ppm).map_err(err)?;
        }
        let wav = dir.join("audio.wav");
        if let Some((pcm, sr, ch)) = audio {
            std::fs::write(&wav, crate::music::wav_bytes(pcm, sr, ch)).map_err(err)?;
        }
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-y"]).arg("-framerate").arg(format!("{fps}")).arg("-i").arg(dir.join("f%05d.ppm"));
        if audio.is_some() {
            cmd.arg("-i").arg(&wav).args(["-c:a", "aac", "-b:a", "192k", "-shortest"]);
        }
        let status = cmd.args(["-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]).arg(out).status().map_err(|e| format!("ffmpeg: {e}"))?;
        if !status.success() {
            return Err("ffmpeg завершился с ошибкой".into());
        }
        Ok(())
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}


/// Компоненты LTX-2.3 из библиотеки.
#[derive(Clone, Debug, Default)]
pub struct LtxPaths {
    /// Чекпойнт (DiT + VAE + вокодер + проекции).
    pub model: PathBuf,
    /// Gemma-3 LTX (текстовый энкодер).
    pub gemma: PathBuf,
    /// Distilled LoRA (dev-чекпойнту для дистиллированных сигм).
    pub distilled_lora: Option<PathBuf>,
    /// Апскейлер ×2 (две стадии, LipDub).
    pub upscaler: Option<PathBuf>,
    pub ic_union: Option<PathBuf>,
    pub ic_motion: Option<PathBuf>,
    pub ic_lipdub: Option<PathBuf>,
}

enum LtxMode {
    Text,
    Frames { first: Option<PathBuf>, last: Option<PathBuf> },
    Retake { video: PathBuf, start: f64, end: f64 },
    Extend { video: PathBuf, last_frame: PathBuf },
    A2v { audio: PathBuf, image: Option<PathBuf> },
    Ic { video: PathBuf, canny: bool },
    Lipdub { video: PathBuf, audio: PathBuf },
}

fn scratch(tag: &str) -> R<PathBuf> {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("synaptix_gen_{tag}_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(err)?;
    Ok(dir)
}

fn ffmpeg_f32(path: &Path, rate: u32, channels: u32) -> R<Vec<f32>> {
    let out = Command::new("ffmpeg")
        .args(["-v", "error", "-y", "-i"])
        .arg(path)
        .args(["-vn", "-ar", &rate.to_string(), "-ac", &channels.to_string(), "-f", "f32le", "-"])
        .output()
        .map_err(|e| format!("ffmpeg: {e}"))?;
    if !out.status.success() {
        return Err(format!("ffmpeg: звук {} не прочитан", path.display()));
    }
    Ok(out.stdout.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Видео → `[1,3,F,H,W]` в [−1,1] (ffmpeg, кадры по fps).
fn load_video_frames(path: &Path, ph: usize, pw: usize, n: usize, fps: f64, dev: Device) -> R<Tensor> {
    load_video_frames_vf(path, ph, pw, n, fps, dev, "")
}

/// То же с дополнительным фильтром ffmpeg после масштаба (`edgedetect` — контуры).
fn load_video_frames_vf(path: &Path, ph: usize, pw: usize, n: usize, fps: f64, dev: Device, extra: &str) -> R<Tensor> {
    let dir = scratch("vin")?;
    let result = (|| -> R<Tensor> {
        let ok = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(path)
            .args(["-vf", &format!("fps={fps},scale={pw}:{ph}:force_original_aspect_ratio=increase,crop={pw}:{ph}{extra}")])
            .args(["-frames:v", &n.to_string()])
            .arg(dir.join("f%05d.png"))
            .status()
            .map_err(|e| format!("ffmpeg: {e}"))?
            .success();
        if !ok {
            return Err(format!("ffmpeg: видео {} не прочитано", path.display()));
        }
        let mut frames = Vec::with_capacity(n);
        for i in 1..=n {
            let p = dir.join(format!("f{i:05}.png"));
            if !p.exists() {
                break;
            }
            let img = synaptix_io::image::load_image(&p, dev).map_err(err)?;
            frames.push(img.contiguous().map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, ph, pw]).map_err(err)?);
        }
        if frames.is_empty() {
            return Err(format!("{}: видео не дало кадров", path.display()));
        }
        // Короче нужного — последний кадр повторяется.
        while frames.len() < n {
            let last = frames.last().unwrap().clone();
            frames.push(last);
        }
        let refs: Vec<&Tensor> = frames.iter().collect();
        Tensor::cat(&refs, 2).map_err(err)?.contiguous().map_err(err)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// Во сколько раз уменьшен образец IC-LoRA при обучении (`…-ref0.5` → 2).
fn ic_downscale(lora: Option<&Path>) -> usize {
    let name = lora.and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    match name.split("ref").nth(1).and_then(|r| r.trim_end_matches(".syn").trim_end_matches(".safetensors").parse::<f64>().ok()) {
        Some(f) if f > 0.0 && f < 1.0 => (1.0 / f).round().max(1.0) as usize,
        _ => 1,
    }
}

/// Длительность медиафайла, с (ffprobe).
fn media_seconds(path: &Path) -> Option<f64> {
    let out = Command::new("ffprobe").args(["-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1"]).arg(path).output().ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

/// LTX-2.3: видео со звуком (дистиллированные сигмы, NAG по умолчанию) — как
/// `synaptix video`: по описанию, с первого и/или последнего кадра, по звуку
/// (липсинк), retake отрезка, продление ролика, IC-LoRA (контуры/движение),
/// LipDub; одна стадия или две (апскейлер ×2 и доводка).
pub fn ltx(paths: &LtxPaths, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
    use synaptix_core::precision::PrecisionConfig;
    use synaptix_llm_gemma3::pipeline::GemmaPipeline;
    use synaptix_video_ltx23::audio_vae::{ltx_log_mel, AudioVaeDecoder, AudioVaeEncoder};
    use synaptix_video_ltx23::dit::AvDit;
    use synaptix_video_ltx23::loader::{LoraWeights, LtxCheckpoint};
    use synaptix_video_ltx23::pipeline as lp;
    use synaptix_video_ltx23::text_encoder::{AudioTextConditioner, VideoTextConditioner};
    use synaptix_video_ltx23::upscaler::Upsampler;
    use synaptix_video_ltx23::vae::{VaeDecoder, VaeEncoder};
    use synaptix_video_ltx23::vocoder::VocoderWithBwe;
    if !device.is_cuda() {
        return Err("LTX-2.3 работает только на видеокарте".into());
    }
    if !req.refs.is_empty() {
        return Err("референсы-картинки понимает MiniMax-H3; у LTX — кадры, звук, видео-образец".into());
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let ord = if let Device::Cuda(o) = device { o } else { 0 };
    let release = || {
        let _ = synaptix_core::device::cuda::synchronize_all(ord);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
    };
    let fps = 24.0;
    let compute = DType::BF16;
    let quant_dit = quant_setting(req.quant.as_deref()).unwrap_or(compute);
    let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(quant_dit.is_quantized());
    let quant_enc = if quant_dit == DType::NVFP4 { DType::NVFP4 } else { DType::MXFP8 };
    let seed = Some(req.seed.max(1));

    // Режим и источник длительности/размера.
    let mode = match (&req.control_video, &req.audio, &req.source_video) {
        (Some(v), Some(a), _) => LtxMode::Lipdub { video: v.clone(), audio: a.clone() },
        (Some(v), None, _) => LtxMode::Ic { video: v.clone(), canny: req.control.as_deref() != Some("motion") },
        (None, _, Some(v)) if req.extend => {
            let dir = scratch("tail")?;
            let frame = dir.join("last.png");
            let ok = Command::new("ffmpeg").args(["-v", "error", "-y", "-sseof", "-0.2", "-i"]).arg(v).args(["-frames:v", "1", "-update", "1"]).arg(&frame).status().map(|s| s.success()).unwrap_or(false);
            if !ok || !frame.exists() {
                return Err(format!("{}: последний кадр не извлечён", v.display()));
            }
            LtxMode::Extend { video: v.clone(), last_frame: frame }
        }
        (None, _, Some(v)) => {
            let (a, b) = req.retake.unwrap_or((0.0, media_seconds(v).unwrap_or(5.0)));
            LtxMode::Retake { video: v.clone(), start: a, end: b }
        }
        (None, Some(a), None) => LtxMode::A2v { audio: a.clone(), image: req.first_frame.clone() },
        (None, None, None) if req.first_frame.is_some() || req.last_frame.is_some() => LtxMode::Frames { first: req.first_frame.clone(), last: req.last_frame.clone() },
        _ => LtxMode::Text,
    };
    let adapter = match &mode {
        LtxMode::Lipdub { .. } => Some(paths.ic_lipdub.clone().ok_or("нет IC-LoRA LipDub (ltx-2.3-…-ic-lora-lipdub)")?),
        LtxMode::Ic { canny: true, .. } => Some(paths.ic_union.clone().ok_or("нет IC-LoRA union-control")?),
        LtxMode::Ic { canny: false, .. } => Some(paths.ic_motion.clone().ok_or("нет IC-LoRA motion-track")?),
        _ => None,
    };
    let distilled_ckpt = paths.model.file_name().map(|n| n.to_string_lossy().to_lowercase().contains("distilled")).unwrap_or(false);
    if adapter.is_some() && !distilled_ckpt {
        return Err("IC-LoRA и LipDub идут на distilled-чекпойнте LTX-2.3 (ltx-2-3-22b-distilled)".into());
    }
    // Retake — одной стадией на целевом разрешении; LipDub — всегда две.
    let two_stage = (req.two_stage && !matches!(mode, LtxMode::Retake { .. })) || matches!(mode, LtxMode::Lipdub { .. });
    if two_stage && paths.upscaler.is_none() {
        return Err("для двух стадий нужен апскейлер LTX-2.3 ×2".into());
    }
    // Длительность: у звука, retake и образцов — по исходнику.
    let src_secs = match &mode {
        LtxMode::A2v { audio, .. } | LtxMode::Lipdub { audio, .. } => media_seconds(audio),
        LtxMode::Retake { video, .. } | LtxMode::Ic { video, .. } => media_seconds(video),
        _ => None,
    };
    // От звука (липсинк, LipDub) длина ролика — по звуку; иначе — запрошенная.
    let secs = match &mode {
        LtxMode::A2v { .. } | LtxMode::Lipdub { .. } => src_secs.or(req.seconds),
        _ => req.seconds.filter(|s| *s > 0.0).or(src_secs),
    }
    .unwrap_or(5.0)
    .clamp(1.0, 20.0);
    let width = req.width.filter(|&v| v > 0).unwrap_or(960);
    let height = req.height.filter(|&v| v > 0).unwrap_or(544);
    let (hp, wp) = lp::latent_grid(width, height);
    let fp = lp::frames_for_duration(secs, fps);
    let out_frames = lp::out_frame_count(fp);

    let ckpt = LtxCheckpoint::open(&paths.model, Device::Cpu, DType::BF16).map_err(|e| format!("LTX: {e}"))?;
    let offload = quant_dit == compute || {
        let dit_b = synaptix_video_ltx23::dit::dit_resident_bytes(&ckpt, quant_dit, compute);
        let vae_b: usize = ckpt.infos().filter(|(n, _, _)| n.starts_with("vae.")).map(|(_, _, s)| compute.bytes_for_numel(s.iter().product())).sum();
        let act_b = fp * hp * wp * 4096 * 2 * 28;
        let pad_b = (fp * hp * wp + 256) * (16384 * 2 + 16384 * 2 + 12288 * 2);
        let need = dit_b + vae_b + act_b + pad_b + (1usize << 30);
        synaptix_core::device::cuda::mem_info(ord).map(|(free, _)| need > free).unwrap_or(true)
    };
    let pin = (offload && quant_dit == compute).then(|| synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused(&ckpt.shard_bytes()));
    let lora_path = adapter.clone().or_else(|| if distilled_ckpt { None } else { paths.distilled_lora.clone() });

    // Текст: Gemma (выгружается) → коннекторы.
    synaptix_core::device::cuda::set_offload_pinned(true);
    let (states, mask, nag_states) = {
        let prec = PrecisionConfig { compute, attn_w: quant_enc, mlp_w: quant_enc, lm_head: DType::BF16, embed: DType::BF16, kv: DType::BF16 };
        let gemma = GemmaPipeline::load_with_precision(&paths.gemma, device, prec, Some(1024)).map_err(|e| format!("Gemma: {e}"))?;
        if let Some(p) = &pin {
            p.resume();
        }
        let (s, m) = gemma.encode_for_ltx(&req.prompt, 1024, device).map_err(|e| format!("Gemma: {e}"))?;
        let nag = gemma.encode_for_ltx(lp::DEFAULT_NAG_PROMPT, 1024, device).map_err(|e| format!("Gemma: {e}"))?;
        (s, m, nag)
    };
    release();
    let (v_enc, a_enc, nag_enc) = {
        let ckpt_gpu = ckpt.view_on(device);
        let vtc = VideoTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?;
        let v = vtc.forward(&states, &mask).map_err(err)?;
        let a = AudioTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?.forward(&states, &mask).map_err(err)?;
        let n = vtc.forward(&nag_states.0, &nag_states.1).map_err(err)?;
        (v, a, n)
    };
    drop(states);
    drop(nag_states);
    synaptix_core::device::cuda::set_offload_pinned(false);
    let v_nag = Some((&nag_enc, lp::NAG_DEFAULT_SCALE, lp::NAG_DEFAULT_ALPHA, lp::NAG_DEFAULT_TAU));

    let total_steps = (lp::DISTILLED_SIGMAS.len() - 1) + if two_stage { lp::STAGE2_SIGMAS.len() - 1 } else { 0 };
    let done = std::sync::atomic::AtomicUsize::new(0);
    let on_step = |_: lp::DenoiseProgress| {
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        progress(n.min(total_steps), total_steps);
    };
    let hooks = lp::DenoiseHooks { progress: Some(&on_step), cancel: Some(cancel) };

    let (latent, a_tok, input_wave) = synaptix_core::grad::no_grad(|| -> R<(Tensor, Option<Tensor>, Option<Vec<f32>>)> {
        let lora = match &lora_path {
            Some(p) => Some(LoraWeights::open(p, device, 1.0).map_err(|e| format!("LoRA {}: {e}", p.display()))?),
            None => None,
        };
        let lora_ckpt = lora.map(|l| ckpt.view_on(Device::Cpu).with_lora(std::sync::Arc::new(l)));
        let dit = AvDit::load_with(lora_ckpt.as_ref().unwrap_or(&ckpt), device, compute, quant_dit, offload).map_err(|e| format!("DiT: {e}"))?;
        let (hp1, wp1) = if two_stage { lp::stage1_grid(hp, wp) } else { (hp, wp) };
        let (hp2, wp2) = (hp1 * 2, wp1 * 2);
        let venc = if matches!(mode, LtxMode::Text | LtxMode::A2v { image: None, .. }) { None } else { Some(VaeEncoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?) };
        let image_tokens = |img: &Tensor, g: (usize, usize)| -> R<Tensor> {
            let (ph, pw) = (g.0 * 32, g.1 * 32);
            let x = synaptix_io::image::fit_image(img, pw, ph, true).map_err(err)?.to_device(device).map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, ph, pw]).map_err(err)?;
            let lat = venc.as_ref().unwrap().encode(&x).map_err(|e| format!("VAE: {e}"))?;
            lp::frame_latent_to_tokens(&lat).map_err(err)
        };
        let latent_tokens = |lat: &Tensor| -> R<(Tensor, usize, usize, usize)> {
            let (f, h, w) = (lat.dims()[2], lat.dims()[3], lat.dims()[4]);
            Ok((lat.reshape(vec![1, 128, f * h * w]).map_err(err)?.transpose(1, 2).map_err(err)?.contiguous().map_err(err)?, f, h, w))
        };
        // Этап денойза на сетке g (stage1: с NAG, без init; stage2: init — апскейл).
        let stage = |g: (usize, usize), sigmas: &[f64], init: Option<(&Tensor, &Tensor)>| -> R<(Tensor, Tensor, Option<Vec<f32>>)> {
            let (vi, ai) = match init {
                Some((v, a)) => (Some(v), Some(a)),
                None => (None, None),
            };
            let nag = if init.is_none() { v_nag } else { None };
            Ok(match &mode {
                LtxMode::Text => {
                    let (l, a) = lp::denoise_av(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, fps, device, nag, &[], seed, &hooks).map_err(err)?;
                    (l, a, None)
                }
                LtxMode::Frames { .. } | LtxMode::Extend { .. } => {
                    let (first, last) = match &mode {
                        LtxMode::Extend { last_frame, .. } => (Some(last_frame.clone()), None),
                        LtxMode::Frames { first, last } => (first.clone(), last.clone()),
                        _ => (None, None),
                    };
                    let load = |p: &PathBuf| synaptix_io::image::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()));
                    let conds = match &first {
                        Some(p) => vec![(0usize, image_tokens(&load(p)?, g)?, 1.0f32)],
                        None => Vec::new(),
                    };
                    match &last {
                        // Последний кадр — ключевым кадром в конце ролика.
                        Some(p) => {
                            let toks = image_tokens(&load(p)?, g)?;
                            let frame = out_frames - 1;
                            let pos = lp::keyframe_positions(g.0, g.1, frame, fps);
                            if let Some((_, first_toks, _)) = conds.first() {
                                // Первый тоже ключевым: два кадра одним блоком.
                                let ft = first_toks.clone();
                                let both = Tensor::cat(&[&ft, &toks], 1).map_err(err)?.contiguous().map_err(err)?;
                                let t = g.0 * g.1;
                                let pos_all = lp::cat_positions(&lp::keyframe_positions(g.0, g.1, 0, fps), t, &pos, t);
                                let (l, a) = lp::denoise_av_append(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, false, Some((&both, &pos_all, 1.0)), None, fps, device, seed, &hooks).map_err(err)?;
                                (l, a, None)
                            } else {
                                let (l, a) = lp::denoise_av_append(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, false, Some((&toks, &pos, 1.0)), None, fps, device, seed, &hooks).map_err(err)?;
                                (l, a, None)
                            }
                        }
                        None => {
                            let (l, a) = lp::denoise_av(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, fps, device, nag, &conds, seed, &hooks).map_err(err)?;
                            (l, a, None)
                        }
                    }
                }
                LtxMode::Retake { video, start, end } => {
                    let frames = load_video_frames(video, g.0 * 32, g.1 * 32, out_frames, fps, device)?;
                    let source = venc.as_ref().unwrap().encode(&frames).map_err(|e| format!("VAE: {e}"))?;
                    let (f_, h_, w_) = (source.dims()[2], source.dims()[3], source.dims()[4]);
                    let (l, a) = lp::denoise_av_retake(&dit, &v_enc, &a_enc, f_, h_, w_, sigmas, &source, *start, *end, fps, device, seed, &hooks).map_err(err)?;
                    (l, a, None)
                }
                LtxMode::A2v { audio, image } => {
                    let fa = lp::audio_token_count(fp, fps);
                    let aenc = AudioVaeEncoder::load(&ckpt, device).map_err(|e| format!("VAE звука: {e}"))?;
                    let mel = ltx_log_mel(&[ffmpeg_f32(audio, 16000, 1)?], device).map_err(err)?;
                    let enc = aenc.encode(&mel).map_err(err)?;
                    drop(aenc);
                    let audio_lat = fit_audio_tokens(enc, fa)?;
                    // Фото — ключевым кадром в начале: говорит именно этот человек/персонаж.
                    let key = match image {
                        Some(p) => {
                            let img = synaptix_io::image::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
                            Some((image_tokens(&img, g)?, lp::keyframe_positions(g.0, g.1, 0, fps)))
                        }
                        None => None,
                    };
                    let v_app = key.as_ref().map(|(t, pos)| (t, pos.as_slice(), 1.0f32));
                    let (l, _) = lp::denoise_av_append(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, Some(&audio_lat), true, v_app, None, fps, device, seed, &hooks).map_err(err)?;
                    (l, audio_lat, Some(ffmpeg_f32(audio, 48000, 2)?))
                }
                LtxMode::Ic { video, canny } => {
                    // Образец — в масштабе, на котором обучена IC-LoRA («ref0.5» — вдвое меньше);
                    // контуры считает ffmpeg (edgedetect) при чтении кадров.
                    let ds = ic_downscale(adapter.as_deref());
                    let edges = if *canny { ",edgedetect=low=0.1:high=0.3,format=rgb24" } else { "" };
                    let frames = load_video_frames_vf(video, g.0 * 32 / ds, g.1 * 32 / ds, out_frames, fps, device, edges)?;
                    let lat = venc.as_ref().unwrap().encode(&frames).map_err(|e| format!("VAE: {e}"))?;
                    let (tok, f_, h_, w_) = latent_tokens(&lat)?;
                    let pos = lp::ref_video_positions(f_, h_, w_, fps, ds);
                    let (l, a) = lp::denoise_av_append(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, false, Some((&tok, &pos, 1.0)), None, fps, device, seed, &hooks).map_err(err)?;
                    (l, a, None)
                }
                LtxMode::Lipdub { video, audio } => {
                    let frames = load_video_frames(video, g.0 * 32, g.1 * 32, out_frames, fps, device)?;
                    let lat = venc.as_ref().unwrap().encode(&frames).map_err(|e| format!("VAE: {e}"))?;
                    let (tok, f_, h_, w_) = latent_tokens(&lat)?;
                    let pos = lp::pixel_coords(f_, h_, w_, fps);
                    // Новая речь — основной звук, ЗАМОРОЖЕННЫЙ на обеих стадиях: губы
                    // строятся под неё (кросс-внимание звук→видео при sigma звука 0).
                    // Раньше на stage1 звук генерировался заново по промпту, губы
                    // шли за ним, а сверху накладывалась речь пользователя — губы
                    // не совпадали и копировали исходник. Звуковой референс — звук
                    // исходного ролика (тембр, как в Dub-It эталона), без него — сама речь.
                    let fa = lp::audio_token_count(fp, fps);
                    let (target, aref) = match ai {
                        Some(a) => (a.clone(), a.clone()),
                        None => {
                            let aenc = AudioVaeEncoder::load(&ckpt, device).map_err(|e| format!("VAE звука: {e}"))?;
                            let enc = |p: &Path| -> R<Tensor> { aenc.encode(&ltx_log_mel(&[ffmpeg_f32(p, 16000, 1)?], device).map_err(err)?).map_err(err) };
                            let target = fit_audio_tokens(enc(audio)?, fa)?;
                            let orig = if has_audio_stream(video) { enc(video).ok() } else { None };
                            drop(aenc);
                            let aref = orig.unwrap_or_else(|| target.clone());
                            (target, aref)
                        }
                    };
                    let apos = lp::lipdub_audio_ref_positions(aref.dims()[1]);
                    let (l, _) = lp::denoise_av_append(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, Some(&target), true, Some((&tok, &pos, 1.0)), Some((&aref, &apos)), fps, device, seed, &hooks).map_err(err)?;
                    (l, target, Some(ffmpeg_f32(audio, 48000, 2)?))
                }
            })
        };
        let (mut l, mut a, mut wave) = stage((hp1, wp1), &lp::DISTILLED_SIGMAS, None)?;
        if two_stage {
            let view = ckpt.view_on(device);
            let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
            let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
            let up = Upsampler::load(paths.upscaler.as_ref().unwrap(), &mean, &std, device).map_err(|e| format!("апскейлер: {e}"))?;
            let l2 = up.upsample(&l).map_err(err)?.to_dtype(compute).map_err(err)?;
            drop(up);
            synaptix_core::tensor::ops::conv_filter_cache_clear();
            let (l3, a3, w3) = stage((hp2, wp2), &lp::STAGE2_SIGMAS, Some((&l2, &a)))?;
            l = l3;
            a = a3;
            if w3.is_some() {
                wave = w3;
            }
        }
        drop(dit);
        drop(venc);
        synaptix_core::tensor::ops::conv_filter_cache_gc();
        Ok((l, Some(a), wave))
    })?;
    release();
    let rgb = synaptix_core::grad::no_grad(|| -> R<Tensor> {
        let vae = VaeDecoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?;
        vae.decode(&latent).map_err(|e| format!("VAE: {e}"))
    })?;
    release();
    // Звук: исходный (липсинк, LipDub) или сгенерированный.
    let pcm: Vec<f32> = match input_wave {
        Some(w) => w,
        None => {
            let wave = synaptix_core::grad::no_grad(|| -> R<Tensor> {
                let avae = AudioVaeDecoder::load(&ckpt, device).map_err(|e| format!("VAE звука: {e}"))?;
                let voc = VocoderWithBwe::load(&paths.model, device).map_err(|e| format!("вокодер: {e}"))?;
                lp::decode_audio_tokens(&avae, &voc, a_tok.as_ref().unwrap()).map_err(err)
            })?;
            release();
            let (ch, l) = (wave.dims()[1], wave.dims()[2]);
            let planar: Vec<f32> = wave.reshape(vec![ch * l]).map_err(err)?.to_dtype(DType::F32).map_err(err)?.to_vec1::<f32>().map_err(err)?;
            (0..l).flat_map(|i| (0..ch).map(move |c| (c, i))).map(|(c, i)| planar[c * l + i]).collect()
        }
    };
    let frames = lp::rgb_to_frames(&rgb).map_err(err)?;
    let n = frames.len();
    let label_mode = match &mode {
        LtxMode::Text => "по описанию",
        LtxMode::Frames { first: Some(_), last: Some(_) } => "между первым и последним кадром",
        LtxMode::Frames { last: Some(_), .. } => "к последнему кадру",
        LtxMode::Frames { .. } => "с первого кадра",
        LtxMode::Retake { .. } => "retake отрезка",
        LtxMode::Extend { .. } => "продление",
        LtxMode::A2v { image: Some(_), .. } => "по звуку с первого кадра (липсинк)",
        LtxMode::A2v { .. } => "по звуку (липсинк)",
        LtxMode::Ic { canny: true, .. } => "по контурам образца (IC-LoRA)",
        LtxMode::Ic { .. } => "по движению образца (IC-LoRA)",
        LtxMode::Lipdub { .. } => "LipDub",
    };
    match &mode {
        LtxMode::Extend { video, last_frame } => {
            // Новое продолжение — следом за исходным роликом (без повтора стыкового кадра).
            let dir = scratch("ext")?;
            let part = dir.join("part.mp4");
            write_mp4(&frames[1..], Some((&pcm, 48_000, 2)), fps, &part)?;
            let ok = Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-i"])
                .arg(video)
                .arg("-i")
                .arg(&part)
                .args(["-filter_complex", &format!("[0:v]fps={fps},scale={}:{},setsar=1[a];[1:v]setsar=1[b];[a][b]concat=n=2:v=1:a=0[v];[0:a][1:a]concat=n=2:v=0:a=1[s]", wp * 32, hp * 32), "-map", "[v]", "-map", "[s]", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart"])
                .arg(out)
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                // У исходника нет звука — склейка только видео + звук продолжения.
                let ok2 = Command::new("ffmpeg")
                    .args(["-v", "error", "-y", "-i"])
                    .arg(video)
                    .arg("-i")
                    .arg(&part)
                    .args(["-filter_complex", &format!("[0:v]fps={fps},scale={}:{},setsar=1[a];[1:v]setsar=1[b];[a][b]concat=n=2:v=1:a=0[v]", wp * 32, hp * 32), "-map", "[v]", "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
                    .arg(out)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                if !ok2 {
                    let _ = std::fs::remove_dir_all(&dir);
                    return Err("ffmpeg: склейка продления не удалась".into());
                }
            }
            let _ = std::fs::remove_dir_all(&dir);
            let _ = std::fs::remove_file(last_frame);
        }
        _ => write_mp4(&frames, Some((&pcm, 48_000, 2)), fps, out)?,
    }
    let label = format!("LTX-2.3 {}×{}, {:.1} с, {label_mode}{}", wp * 32, hp * 32, n as f64 / fps, if two_stage { ", две стадии" } else { "" });
    Ok(VideoOutput { width: wp * 32, height: hp * 32, frames: n, fps, label })
}

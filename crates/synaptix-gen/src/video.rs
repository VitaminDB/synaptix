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

/// Формат весов, которым квантован бандл (`quant_manifest.json`): у
/// квантованного файла DiT берёт его же формат — перекодировка в другой
/// квант (MXFP8 → NVFP4) теряет качество, а плотное чтение деквантует
/// и стримит вдвое больший вес.
pub(crate) fn stored_format(model: &Path) -> Option<DType> {
    let b = synaptix_bundle::Bundle::open(model).ok()?;
    let raw = b.read_file("quant_manifest.json").ok()?;
    let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    let fmt = v.get("tensors")?.as_object()?.values().find_map(|t| t.get("format")?.as_str().map(str::to_string))?;
    quant_setting(Some(&fmt))
}

/// Квант из настройки; `None` — умолчание пайплайна.
pub(crate) fn quant_setting(q: Option<&str>) -> Option<DType> {
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

/// H3: видео в `out` (mp4). `progress(шаг, всего)`. Стадии
/// [`crate::h3_stages`] подряд: энкодер → (VAE ключевых кадров) → DiT →
/// VAE видео → VAE звука, между этапами пулы ужимаются.
pub fn h3(model: &Path, encoder: Option<&Path>, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
    use crate::h3_stages as st;
    let m = st::open(model, encoder, device, req.quant.as_deref())?;
    let size = st::Size { width: req.width.unwrap_or(0), height: req.height.unwrap_or(0), seconds: req.seconds.unwrap_or(0.0) };
    let steps = req.steps.filter(|&s| s > 0).unwrap_or(20);
    let cfg_scale = req.guidance.unwrap_or(5.0);
    // Негатив — только при CFG > 1 (без CFG его не кодировать).
    let negative = req.negative.as_deref().filter(|_| cfg_scale > 1.0);
    let cond = m.encode(size, &req.prompt, negative, req.first_frame.as_deref(), req.last_frame.as_deref(), &req.refs, st::RefSize::Match)?;
    let keyframes = req.first_frame.is_some() as usize + req.last_frame.is_some() as usize;
    let lat = m.sample(&cond, steps, cfg_scale, req.seed, req.source_video.as_deref(), req.strength, cancel, progress)?;
    drop(cond);
    let frames = m.decode_video(&lat)?;
    let (pcm, sample_rate, channels) = m.decode_audio(&lat)?;
    write_mp4(&frames, Some((&pcm, sample_rate, channels)), st::FPS, out)?;
    let (width, height) = (lat.width, lat.height);
    let label = format!("MiniMax-H3 {width}×{height}, {:.1} с, {steps} шагов{}", lat.frames as f64 / st::FPS, if keyframes == 0 { String::new() } else { format!(", ключевых кадров {keyframes}") }) + &if req.refs.is_empty() { String::new() } else { format!(", референсов {}", req.refs.len()) };
    Ok(VideoOutput { width, height, frames: lat.frames, fps: st::FPS, label })
}

/// Кадры `[3,H,W]` 0..1 и звук (перемежённые отсчёты) → mp4 через ffmpeg.
/// Звуковые токены LTX ровно на `fa` кадров: обрезка или повтор последнего.
pub(crate) fn fit_audio_tokens(enc: Tensor, fa: usize) -> R<Tensor> {
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

pub(crate) fn ffmpeg_f32(path: &Path, rate: u32, channels: u32) -> R<Vec<f32>> {
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
pub(crate) fn media_seconds(path: &Path) -> Option<f64> {
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
    let quant_dit = quant_setting(req.quant.as_deref()).or_else(|| stored_format(&paths.model)).unwrap_or(compute);
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

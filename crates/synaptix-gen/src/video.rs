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
    /// LTX-2.5: удвоить частоту кадров временным апскейлером латента (24 → 48 кадр/с).
    pub fps_x2: bool,
    /// LTX-2.5: декодировать диффузионным видеодекодером (резче лица, текстуры, текст; дольше).
    pub diff_decoder: bool,
    /// LTX-2.5: Diffusion Fidelity Rendering — половинное разрешение со слотами
    /// ключевых кадров, затем полное с IC-LoRA детализации.
    pub dfr: bool,
    /// H3 restyle: сила перерисовки `source_video` (0..1; по умолчанию 0.6).
    pub strength: Option<f32>,
    /// Латент хвоста предыдущего ролика (`.safetensors`): продолжить с хвоста без PNG.
    pub context: Option<PathBuf>,
    /// Сколько латентных кадров сохранять в хвост (0 / None — умолчание модели).
    pub context_frames: Option<usize>,
    /// Куда записать `.safetensors` латента хвоста этого ролика (до декодирования VAE).
    pub tail_out: Option<PathBuf>,
    /// Идентификатор модели в службе (пишется в метаданные хвоста и сверяется при продолжении).
    pub model_id: Option<String>,
    /// H3: LoRA поверх DiT (Turbo и др.) и её сила (по умолчанию 1).
    /// У Turbo без явных steps/guidance — шаги по имени файла («8step», «4step»), CFG 1.
    pub lora: Option<PathBuf>,
    pub lora_strength: Option<f32>,
    /// LTX, полный (не distilled) чекпойнт: управляемая генерация как у Lightricks
    /// (`ti2vid_two_stages`: стадия 1 в половинном разрешении — полные шаги с
    /// CFG/STG/модальностями и отрицательным промптом, апскейл ×2, стадия 2 —
    /// дистилл-LoRA). `None` — включено для текста и первого кадра; `Some(false)` —
    /// прежний быстрый путь (дистилл-LoRA на обеих стадиях).
    pub guided: Option<bool>,
    /// LTX-2.5 IC-LoRA Ingredients: лист-референс (одна картинка: каждый герой —
    /// крупный план и развороты, предметы и место — своими панелями, чёрный фон,
    /// без текста) — герои, предметы и место держатся по нему. Промпт:
    /// «Reference sheet: <панели> / Generated video: <действие>».
    pub ref_sheet: Option<PathBuf>,
}

/// Отрицательный промпт Lightricks по умолчанию (`DEFAULT_NEGATIVE_PROMPT` ltx-pipelines).
pub const LTX_NEGATIVE_PROMPT: &str = "has_subtitles, has_blurbox, transition from black, transition to black, speech_ending_short, \
blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, \
grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, \
deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, \
wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of \
field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent \
lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny \
valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, \
mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, \
off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward \
pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, \
inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts.";

/// Отрицательный промпт из карточки IC-LoRA Ingredients.
pub const INGREDIENTS_NEGATIVE_PROMPT: &str = "worst quality, inconsistent motion, blurry, jittery, distorted";

/// Гайдеры видео и звука по умолчанию Lightricks для полного чекпойнта (2.3+:
/// STG на блоке 28; 2.0 — 29); `cfg` — своя сила CFG видео.
fn guided_params(version: &str, cfg: Option<f32>) -> (synaptix_video_ltx23::guider::GuiderParams, synaptix_video_ltx23::guider::GuiderParams) {
    use synaptix_video_ltx23::guider::GuiderParams;
    let block = if ltx_version_at_least(version, 2, 3) { 28 } else { 29 };
    let v = GuiderParams { cfg_scale: cfg.filter(|c| *c > 0.0).unwrap_or(3.0), stg_scale: 1.0, rescale_scale: 0.7, modality_scale: 3.0, skip_step: 0, stg_blocks: vec![block] };
    let a = GuiderParams { cfg_scale: 7.0, stg_scale: 1.0, rescale_scale: 0.7, modality_scale: 3.0, skip_step: 0, stg_blocks: vec![block] };
    (v, a)
}

/// Число шагов дистиллированной LoRA по имени файла (`…_8step_…` → 8).
pub fn lora_steps(p: &Path) -> Option<usize> {
    let n = p.file_name()?.to_string_lossy().to_ascii_lowercase();
    let i = n.find("step")?;
    let digits: String = n[..i].chars().rev().take_while(|c| c.is_ascii_digit()).collect::<Vec<_>>().into_iter().rev().collect();
    digits.parse().ok().filter(|&s: &usize| s > 0 && s <= 64)
}

/// Формат весов, которым квантован бандл (`quant_manifest.json`): у
/// квантованного файла DiT берёт его же формат — перекодировка в другой
/// квант (MXFP8 → NVFP4) теряет качество, а плотное чтение деквантует
/// и стримит вдвое больший вес.
/// Чекпойнт LTX-2.5 и новее (сэмплер Euler-ancestral, подпись «LTX-2.5»);
/// `SYN_LTX_SAMPLER=euler` — прежний Euler.
pub(crate) fn is_ltx25(model: &Path) -> bool {
    std::env::var("SYN_LTX_SAMPLER").as_deref() != Ok("euler")
        && synaptix_video_ltx23::loader::LtxCheckpoint::open(model, Device::Cpu, DType::BF16)
            .is_ok_and(|c| ltx_version_at_least(&c.config.model_version, 2, 5))
}

/// Версия LTX из метаданных (`"2.5.0"`) не ниже `major.minor`.
fn ltx_version_at_least(v: &str, major: u32, minor: u32) -> bool {
    let mut it = v.split('.').map(|x| x.trim().parse::<u32>().unwrap_or(0));
    let (a, b) = (it.next().unwrap_or(0), it.next().unwrap_or(0));
    (a, b) >= (major, minor)
}

pub(crate) fn stored_format(model: &Path) -> Option<DType> {
    let b = synaptix_bundle::Bundle::open(model).ok()?;
    let raw = b.read_file("quant_manifest.json").ok()?;
    let v: serde_json::Value = serde_json::from_slice(&raw).ok()?;
    // Формат трансформера (энкодер текста бывает квантован и при плотном DiT).
    let fmt = v
        .get("tensors")?
        .as_object()?
        .iter()
        .filter(|(k, _)| k.contains("transformer_blocks") || k.starts_with("blocks."))
        .find_map(|(_, t)| t.get("format")?.as_str().map(str::to_string))?;
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
    /// Записанный файл латента хвоста (если запрашивался `tail_out`).
    pub tail: Option<PathBuf>,
    /// Ролик был продолжен по латенту хвоста (`context`).
    pub continued_from_latent: bool,
}

/// H3: видео в `out` (mp4). `progress(шаг, всего)`. Стадии
/// [`crate::h3_stages`] подряд: энкодер → (VAE ключевых кадров) → DiT →
/// (хвост в safetensors) → VAE видео → VAE звука, между этапами пулы ужимаются.
pub fn h3(model: &Path, encoder: Option<&Path>, device: Device, req: &VideoRequest, out: &Path, cancel: &AtomicBool, progress: &(dyn Fn(usize, usize) + Sync)) -> R<VideoOutput> {
    use crate::h3_stages as st;
    use crate::tail::TailLatent;
    let mut m = st::open(model, encoder, device, req.quant.as_deref())?;
    let lora = req.lora.as_ref().filter(|_| req.lora_strength.unwrap_or(1.0) > 0.0);
    if let Some(p) = lora {
        if !p.exists() {
            return Err(format!("LoRA нет: {}", p.display()));
        }
        m = m.with_lora(p.clone(), req.lora_strength.unwrap_or(1.0));
    }
    // дистиллированная LoRA (Turbo): свои шаги и без CFG
    let distilled = lora.and_then(|p| lora_steps(p));
    let context = match &req.context {
        Some(p) => Some(TailLatent::load(p)?),
        None => None,
    };
    let continued = context.is_some();
    let ctx_lat_frames = context.as_ref().map(|c| c.meta.context_frames).unwrap_or(0);
    let size = st::Size { width: req.width.unwrap_or(0), height: req.height.unwrap_or(0), seconds: req.seconds.unwrap_or(0.0) };
    let steps = req.steps.filter(|&s| s > 0).or(distilled).unwrap_or(20);
    let cfg_scale = req.guidance.unwrap_or(if distilled.is_some() { 1.0 } else { 5.0 });
    // Негатив — только при CFG > 1 (без CFG его не кодировать).
    let negative = req.negative.as_deref().filter(|_| cfg_scale > 1.0);
    let first = if continued { None } else { req.first_frame.as_deref() };
    let _status = synaptix_core::gen_status::Scope::new("кодирование текста и кадров");
    let cond = m.encode_with_context(size, &req.prompt, negative, first, req.last_frame.as_deref(), &req.refs, st::RefSize::Match, context, req.model_id.as_deref())?;
    let keyframes = first.is_some() as usize + req.last_frame.is_some() as usize;
    synaptix_core::gen_status::phase("загрузка DiT");
    let lat = m.sample(&cond, steps, cfg_scale, req.seed, req.source_video.as_deref(), req.strength, cancel, progress)?;
    drop(cond);
    let tail_path = if let Some(tp) = &req.tail_out {
        let model_name = req.model_id.as_deref().unwrap_or("minimax-h3");
        let tail = lat.tail(req.context_frames.unwrap_or(0), model_name)?;
        tail.save(tp)?;
        Some(tp.clone())
    } else {
        None
    };
    synaptix_core::gen_status::phase("декодер VAE: видео");
    let frames = m.decode_video(&lat)?;
    synaptix_core::gen_status::phase("декодер VAE: звук");
    let (pcm, sample_rate, channels) = m.decode_audio(&lat)?;
    synaptix_core::gen_status::phase("запись mp4");
    write_mp4(&frames, Some((&pcm, sample_rate, channels)), st::FPS, out)?;
    let (width, height) = (lat.width, lat.height);
    let n_frames = frames.len();
    let mut label = format!("MiniMax-H3 {width}×{height}, {:.1} с, {steps} шагов", n_frames as f64 / st::FPS);
    if let Some(p) = lora {
        label += &format!(", LoRA {}", p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    }
    if continued {
        label += &format!(", с хвоста ({ctx_lat_frames} лат.)");
    } else if keyframes > 0 {
        label += &format!(", ключевых кадров {keyframes}");
    }
    if !req.refs.is_empty() {
        label += &format!(", референсов {}", req.refs.len());
    }
    Ok(VideoOutput { width, height, frames: n_frames, fps: st::FPS, label, tail: tail_path, continued_from_latent: continued })
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
    /// LTX-2.5: IC-LoRA Ingredients (лист-референс героев, предметов, места).
    pub ic_ingredients: Option<PathBuf>,
    /// LTX-2.5: голова длительности (длина ролика по описанию, когда она не задана).
    pub duration_head: Option<PathBuf>,
    /// LTX-2.5: временной апскейлер ×2 (удвоение частоты кадров).
    pub temporal_upscaler: Option<PathBuf>,
    /// LTX-2.5: IC-LoRA детализации ×2 (этап 2 DFR).
    pub detail_lora: Option<PathBuf>,
    /// LTX-2.5: диффузионный видеодекодер (вместо свёрточного VAE).
    pub diff_vae: Option<PathBuf>,
}

/// LTX-2.5: длина ролика в секундах по описанию — текстовый кодер, коннекторы
/// и голова длительности (эталон `duration_predictor.py`; пределы 1…20 с,
/// сетка 8k+1 кадров при 24 кадр/с).
pub fn ltx_auto_seconds(paths: &LtxPaths, device: Device, prompt: &str, quant: Option<&str>) -> R<f64> {
    use synaptix_core::precision::PrecisionConfig;
    use synaptix_video_ltx23::loader::LtxCheckpoint;
    use synaptix_video_ltx23::text_encoder::{AudioTextConditioner, VideoTextConditioner};
    let head_path = paths.duration_head.as_ref().ok_or("у модели нет головы длительности")?;
    let compute = DType::BF16;
    let quant_dit = quant_setting(quant).or_else(|| stored_format(&paths.model)).unwrap_or(compute);
    let quant_enc = if quant_dit == DType::NVFP4 { DType::NVFP4 } else { DType::MXFP8 };
    let ord = if let Device::Cuda(o) = device { o } else { 0 };
    let (states, mask) = {
        let prec = PrecisionConfig { compute, attn_w: quant_enc, mlp_w: quant_enc, lm_head: DType::BF16, embed: DType::BF16, kv: DType::BF16 };
        let enc = crate::ltx_text::LtxTextEncoder::load(&paths.gemma, device, prec, Some(1024))?;
        enc.encode_for_ltx(prompt, 1024, device)?
    };
    let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
    let ckpt = LtxCheckpoint::open(&paths.model, Device::Cpu, DType::BF16).map_err(|e| format!("LTX: {e}"))?;
    let view = ckpt.view_on(device);
    let v = VideoTextConditioner::load(&view, device, compute).map_err(err)?.forward(&states, &mask).map_err(err)?;
    let a = AudioTextConditioner::load(&view, device, compute).map_err(err)?.forward(&states, &mask).map_err(err)?;
    let head = synaptix_video_ltx23::duration::DurationHead::load(head_path, device).map_err(err)?;
    let secs = head.seconds(&v, &a).map_err(err)?;
    let frames = synaptix_video_ltx23::duration::frames_for(secs, 24.0, 1.0, 20.0);
    let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
    Ok(frames as f64 / 24.0)
}

enum LtxMode {
    Text,
    Frames { first: Option<PathBuf>, last: Option<PathBuf> },
    Continue { tail: crate::tail::TailLatent },
    Retake { video: PathBuf, start: f64, end: f64 },
    Extend { video: PathBuf, tail_frames: usize },
    A2v { audio: PathBuf, image: Option<PathBuf> },
    Ic { video: PathBuf, canny: bool },
    Lipdub { video: PathBuf, audio: PathBuf },
    /// Лист-референс IC-LoRA Ingredients.
    Sheet { image: PathBuf },
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
    let mode = if let Some(ctx_path) = &req.context {
        let tail = crate::tail::TailLatent::load(ctx_path)?;
        tail.meta.check_compat("ltx", req.model_id.as_deref(), req.width, req.height, None)?;
        LtxMode::Continue { tail }
    } else if let Some(sheet) = &req.ref_sheet {
        LtxMode::Sheet { image: sheet.clone() }
    } else {
        match (&req.control_video, &req.audio, &req.source_video) {
            (Some(v), Some(a), _) => LtxMode::Lipdub { video: v.clone(), audio: a.clone() },
            (Some(v), None, _) => LtxMode::Ic { video: v.clone(), canny: req.control.as_deref() != Some("motion") },
            (None, _, Some(v)) if req.extend => {
                let t_ctx = req.context_frames.filter(|&n| n > 0).unwrap_or(crate::ltx_stages::DEFAULT_CONTEXT_FRAMES).clamp(1, 5);
                LtxMode::Extend { video: v.clone(), tail_frames: lp::out_frame_count(t_ctx) }
            }
            (None, _, Some(v)) => {
                let (a, b) = req.retake.unwrap_or((0.0, media_seconds(v).unwrap_or(5.0)));
                LtxMode::Retake { video: v.clone(), start: a, end: b }
            }
            (None, Some(a), None) => LtxMode::A2v { audio: a.clone(), image: req.first_frame.clone() },
            (None, None, None) if req.first_frame.is_some() || req.last_frame.is_some() => LtxMode::Frames { first: req.first_frame.clone(), last: req.last_frame.clone() },
            _ => LtxMode::Text,
        }
    };
    let fps = match &mode {
        LtxMode::Continue { tail } => tail.meta.fps,
        _ => fps,
    };
    let adapter = match &mode {
        LtxMode::Lipdub { .. } => Some(paths.ic_lipdub.clone().ok_or("нет IC-LoRA LipDub (ltx-2.3-…-ic-lora-lipdub)")?),
        LtxMode::Ic { canny: true, .. } => Some(paths.ic_union.clone().ok_or("нет IC-LoRA union-control")?),
        LtxMode::Ic { canny: false, .. } => Some(paths.ic_motion.clone().ok_or("нет IC-LoRA motion-track")?),
        LtxMode::Sheet { .. } => Some(paths.ic_ingredients.clone().ok_or("нет IC-LoRA Ingredients (ltx-2.5-…-ic-lora-ingredients)")?),
        _ => None,
    };
    let distilled_ckpt = paths.model.file_name().map(|n| n.to_string_lossy().to_lowercase().contains("distilled")).unwrap_or(false);
    // Лист Ingredients на полном чекпойнте — управляемым путём (dev + IC-LoRA);
    // прочие IC-LoRA и LipDub — как у Lightricks, только на distilled.
    let sheet_on_full = !distilled_ckpt && matches!(mode, LtxMode::Sheet { .. });
    // Лист на полном чекпойнте — по карточке IC-LoRA Ingredients (валидация
    // Lightricks): dev без дистилл-LoRA, ОДНА стадия в полном разрешении
    // (обучено на 768×448×121), 30 шагов, CFG 4, STG 1.0 на блоке 29 (видео),
    // отрицательный промпт карточки. Одна стадия distilled копирует лист, две
    // стадии дают мозаику (так и у сторонних замеров).
    let sheet_dev = sheet_on_full;
    if adapter.is_some() && !distilled_ckpt && !sheet_on_full {
        return Err("IC-LoRA и LipDub идут на distilled-чекпойнте (ltx-2-3-22b-distilled, ltx-2-5-22b-distilled)".into());
    }
    // Retake — одной стадией на целевом разрешении; LipDub — всегда две;
    // Continue с хвостом без stage1_video — одной стадией на точном разрешении хвоста.
    let two_stage = (req.two_stage
        && !matches!(mode, LtxMode::Retake { .. })
        && !matches!(&mode, LtxMode::Continue { tail } if tail.video_stage1.is_none()))
        || matches!(mode, LtxMode::Lipdub { .. });
    let dfr = req.dfr && matches!(mode, LtxMode::Text | LtxMode::Frames { last: None, .. } | LtxMode::Continue { .. });
    if dfr && !distilled_ckpt {
        // Как DFRPipeline Lightricks: дистиллированное расписание на distilled-чекпойнте;
        // на полном стадия 2 шла бы без дистилл-LoRA с дистилл-сигмами.
        return Err("DFR идёт на distilled-чекпойнте (ltx-2-5-22b-distilled); у полного — управляемый путь (guided)".into());
    }
    if dfr && paths.detail_lora.is_none() {
        return Err("DFR: у модели нет IC-LoRA детализации (ltx-2.5-…-ic-lora-pixel-spatial-upscaler)".into());
    }
    // Полный чекпойнт: управляемый путь Lightricks (текст, первый кадр).
    let guided = sheet_dev
        || (!distilled_ckpt
            && adapter.is_none()
            && !dfr
            && req.guided.unwrap_or(true)
            && matches!(mode, LtxMode::Text | LtxMode::Frames { last: None, .. }));
    let two_stage = !sheet_dev && (two_stage || dfr || (guided && paths.upscaler.is_some()));
    if two_stage && paths.upscaler.is_none() {
        return Err("для двух стадий нужен апскейлер LTX-2.3 ×2".into());
    }
    // Длительность: у звука, retake и образцов — по исходнику.
    let src_secs = match &mode {
        LtxMode::A2v { audio, .. } | LtxMode::Lipdub { audio, .. } => media_seconds(audio),
        LtxMode::Retake { video, .. } | LtxMode::Ic { video, .. } => media_seconds(video),
        _ => None,
    };
    // LTX-2.5: длина не задана — по описанию (голова длительности).
    let auto_secs = match &mode {
        LtxMode::Text | LtxMode::Frames { .. } if req.seconds.is_none_or(|s| s <= 0.0) && paths.duration_head.is_some() => {
            let s = ltx_auto_seconds(paths, device, &req.prompt, req.quant.as_deref())?;
            eprintln!("[ltx] длительность по описанию: {s:.2} с");
            Some(s)
        }
        _ => None,
    };
    // От звука (липсинк, LipDub) длина ролика — по звуку; иначе — запрошенная.
    let secs = match &mode {
        LtxMode::A2v { .. } | LtxMode::Lipdub { .. } => src_secs.or(req.seconds),
        _ => req.seconds.filter(|s| *s > 0.0).or(auto_secs).or(src_secs),
    }
    .unwrap_or(5.0)
    .clamp(1.0, 20.0);
    let width = match &mode {
        LtxMode::Continue { tail } => tail.meta.width,
        _ => req.width.filter(|&v| v > 0).unwrap_or(960),
    };
    let height = match &mode {
        LtxMode::Continue { tail } => tail.meta.height,
        _ => req.height.filter(|&v| v > 0).unwrap_or(544),
    };
    let (hp, wp) = lp::latent_grid(width, height);
    let (fp, overlap_frames) = match &mode {
        LtxMode::Continue { tail } => {
            let fp_new = ((secs * fps) / 8.0).round().max(1.0) as usize;
            (tail.meta.context_frames + fp_new, tail.meta.overlap_frames)
        }
        LtxMode::Extend { tail_frames, .. } => {
            let t_ctx = lp::fp_for_frames(*tail_frames);
            let fp_new = ((secs * fps) / 8.0).round().max(1.0) as usize;
            (t_ctx + fp_new, lp::out_frame_count(t_ctx))
        }
        _ => (lp::frames_for_duration(secs, fps), 0),
    };
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
    // Управляемый путь: стадия 1 — чистый полный DiT, дистилл-LoRA — только на стадии 2.
    let lora_path = adapter.clone().or_else(|| if distilled_ckpt || guided { None } else { paths.distilled_lora.clone() });

    // Текст: Gemma (выгружается) → коннекторы.
    let _status = synaptix_core::gen_status::Scope::new("кодирование текста (Gemma)");
    synaptix_core::device::cuda::set_offload_pinned(true);
    let (states, mask, nag_states) = {
        let prec = PrecisionConfig { compute, attn_w: quant_enc, mlp_w: quant_enc, lm_head: DType::BF16, embed: DType::BF16, kv: DType::BF16 };
        let gemma = crate::ltx_text::LtxTextEncoder::load(&paths.gemma, device, prec, Some(1024))?;
        if let Some(p) = &pin {
            p.resume();
        }
        let (s, m) = gemma.encode_for_ltx(&req.prompt, 1024, device).map_err(|e| format!("Gemma: {e}"))?;
        let nag = gemma.encode_for_ltx(lp::DEFAULT_NAG_PROMPT, 1024, device).map_err(|e| format!("Gemma: {e}"))?;
        // Управляемый путь: отрицательный промпт (свой или Lightricks) для CFG.
        let neg = if guided {
            let default = if matches!(mode, LtxMode::Sheet { .. }) { INGREDIENTS_NEGATIVE_PROMPT } else { LTX_NEGATIVE_PROMPT };
            let text = req.negative.as_deref().map(str::trim).filter(|t| !t.is_empty()).unwrap_or(default);
            Some(gemma.encode_for_ltx(text, 1024, device).map_err(|e| format!("Gemma: {e}"))?)
        } else {
            None
        };
        (s, m, (nag, neg))
    };
    let (nag_states, neg_states) = nag_states;
    release();
    let (v_enc, a_enc, nag_enc) = {
        let ckpt_gpu = ckpt.view_on(device);
        let vtc = VideoTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?;
        let v = vtc.forward(&states, &mask).map_err(err)?;
        let a = AudioTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?.forward(&states, &mask).map_err(err)?;
        let n = vtc.forward(&nag_states.0, &nag_states.1).map_err(err)?;
        let neg_enc = match &neg_states {
            Some((s, m)) => Some((vtc.forward(s, m).map_err(err)?, AudioTextConditioner::load(&ckpt_gpu, device, compute).map_err(err)?.forward(s, m).map_err(err)?)),
            None => None,
        };
        (v, a, (n, neg_enc))
    };
    let (nag_enc, neg_enc) = nag_enc;
    drop(states);
    drop(nag_states);
    drop(neg_states);
    synaptix_core::device::cuda::set_offload_pinned(false);
    let v_nag = Some((&nag_enc, lp::NAG_DEFAULT_SCALE, lp::NAG_DEFAULT_ALPHA, lp::NAG_DEFAULT_TAU));

    let guided_steps = req.steps.filter(|s| *s > 0).unwrap_or(30);
    let stage1_steps = if guided { guided_steps } else { lp::DISTILLED_SIGMAS.len() - 1 };
    let total_steps = stage1_steps + if two_stage { lp::STAGE2_SIGMAS.len() - 1 } else { 0 };
    let done = std::sync::atomic::AtomicUsize::new(0);
    let on_step = |_: lp::DenoiseProgress| {
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        progress(n.min(total_steps), total_steps);
    };
    let hooks = lp::DenoiseHooks { progress: Some(&on_step), cancel: Some(cancel) };

    let latent = synaptix_core::grad::no_grad(|| -> R<(Tensor, Option<Tensor>, Option<Vec<f32>>, Option<Tensor>)> {
        // Сила IC-LoRA: 1.0 (карточка 2.5; у 2.3 — 1.4), `SYN_LTX_IC_STRENGTH` — подбор.
        let ic_k = std::env::var("SYN_LTX_IC_STRENGTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0f32);
        let lora = match &lora_path {
            Some(p) => Some(LoraWeights::open(p, device, if adapter.is_some() { ic_k } else { 1.0 }).map_err(|e| format!("LoRA {}: {e}", p.display()))?),
            None => None,
        };
        let lora_ckpt = lora.map(|l| ckpt.view_on(Device::Cpu).with_lora(std::sync::Arc::new(l)));
        synaptix_core::gen_status::phase("загрузка DiT");
        let dit = AvDit::load_with(lora_ckpt.as_ref().unwrap_or(&ckpt), device, compute, quant_dit, offload).map_err(|e| format!("DiT: {e}"))?;
        synaptix_core::gen_status::phase(if offload { "денойз (DiT стримится с хоста)" } else { "денойз" });
        let (hp1, wp1) = if two_stage { lp::stage1_grid(hp, wp) } else { (hp, wp) };
        let (hp2, wp2) = (hp1 * 2, wp1 * 2);
        let venc = if matches!(mode, LtxMode::Text | LtxMode::Continue { .. } | LtxMode::A2v { image: None, .. }) { None } else { Some(VaeEncoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?) };
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
        if guided {
            // Lightricks `ti2vid_two_stages` для полного чекпойнта: стадия 1 в
            // половинном разрешении — `guided_steps` шагов LTX2Scheduler с CFG,
            // STG и изоляцией модальностей (Euler), апскейл ×2, стадия 2 — тот же
            // DiT с дистилл-LoRA, STAGE2_SIGMAS, звук стадии 1 сохраняется.
            synaptix_video_ltx23::runtime::set_ancestral(None);
            let first = match &mode {
                LtxMode::Frames { first: Some(p), .. } => Some(synaptix_io::image::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?),
                _ => None,
            };
            let conds = |g: (usize, usize)| -> R<Vec<(usize, Tensor, f32)>> {
                Ok(match &first {
                    Some(img) => vec![(0usize, image_tokens(img, g)?, 1.0f32)],
                    None => Vec::new(),
                })
            };
            // Лист Ingredients: статичное видео в разрешении стадии (масштаб 1), ≥121 кадра.
            let sheet_ref = |g: (usize, usize)| -> R<Option<(Tensor, Vec<f64>)>> {
                let LtxMode::Sheet { image } = &mode else { return Ok(None) };
                let img = synaptix_io::image::load_image(image, Device::Cpu).map_err(|e| format!("{}: {e}", image.display()))?;
                let (ph, pw) = (g.0 * 32, g.1 * 32);
                let one = synaptix_io::image::fit_image(&img, pw, ph, true).map_err(err)?.to_device(device).map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, ph, pw]).map_err(err)?;
                let n = lp::out_frame_count(lp::fp_for_frames(out_frames.max(121)));
                let copies: Vec<&Tensor> = std::iter::repeat_n(&one, n).collect();
                let frames = Tensor::cat(&copies, 2).map_err(err)?.contiguous().map_err(err)?;
                let lat = venc.as_ref().unwrap().encode(&frames).map_err(|e| format!("VAE: {e}"))?;
                let (tok, f_, h_, w_) = latent_tokens(&lat)?;
                Ok(Some((tok, lp::ref_video_positions(f_, h_, w_, fps, 1))))
            };
            let (mut vg, ag) = guided_params(&ckpt.config.model_version, req.guidance);
            if sheet_dev {
                // Валидация Ingredients: CFG 4, STG 1.0 на блоке 29 (видео).
                vg.cfg_scale = req.guidance.filter(|c| *c > 0.0).unwrap_or(4.0);
                vg.stg_blocks = vec![29];
            }
            let (vn, an) = neg_enc.as_ref().ok_or("LTX: нет отрицательного промпта")?;
            let sig = lp::ltx2_sigmas(guided_steps, fp * hp1 * wp1);
            eprintln!("[ltx] полный чекпойнт: стадия 1 {}×{}, {guided_steps} шагов, CFG видео {} / звук {}, STG {:?}", wp1 * 32, hp1 * 32, vg.cfg_scale, ag.cfg_scale, vg.stg_blocks);
            synaptix_core::gen_status::phase("стадия 1: полный DiT, CFG/STG");
            let ref1 = sheet_ref((hp1, wp1))?;
            let (l1, a1) = lp::denoise_av_guided(&dit, &v_enc, &a_enc, vn, an, &vg, &ag, fp, hp1, wp1, &sig, fps, device, &conds((hp1, wp1))?, ref1.as_ref().map(|(t, p)| (t, p.as_slice())), seed, &hooks).map_err(err)?;
            drop(ref1);
            drop(dit);
            release();
            if !two_stage {
                return Ok((l1, Some(a1), None, None));
            }
            let l2 = {
                let view = ckpt.view_on(device);
                let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
                let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
                let up = Upsampler::load(paths.upscaler.as_ref().unwrap(), &mean, &std, device).map_err(|e| format!("апскейлер: {e}"))?;
                up.upsample(&l1).map_err(err)?.to_dtype(compute).map_err(err)?
            };
            synaptix_core::tensor::ops::conv_filter_cache_clear();
            release();
            synaptix_core::gen_status::phase("стадия 2: загрузка DiT с дистилл-LoRA");
            // Стадия 2: дистилл-LoRA (+ IC-LoRA листа, ΔW складываются).
            let mut view2 = ckpt.view_on(Device::Cpu);
            for p in paths.distilled_lora.iter().chain(adapter.iter()) {
                let l = LoraWeights::open(p, device, 1.0).map_err(|e| format!("LoRA {}: {e}", p.display()))?;
                view2 = view2.with_lora(std::sync::Arc::new(l));
            }
            if paths.distilled_lora.is_none() {
                eprintln!("[ltx] нет дистилл-LoRA — стадия 2 полным DiT");
            }
            let dit2 = AvDit::load_with(&view2, device, compute, quant_dit, offload).map_err(|e| format!("DiT: {e}"))?;
            synaptix_core::gen_status::phase("стадия 2: доводка ×2");
            let l3 = match sheet_ref((hp2, wp2))? {
                Some((tok2, pos2)) => lp::denoise_av_append(&dit2, &v_enc, &a_enc, fp, hp2, wp2, &lp::STAGE2_SIGMAS, Some(&l2), Some(&a1), true, Some((&tok2, &pos2, 1.0)), None, fps, device, seed, &hooks).map_err(err)?.0,
                None => lp::denoise_av_cond(&dit2, &v_enc, &a_enc, fp, hp2, wp2, &lp::STAGE2_SIGMAS, Some(&l2), None, fps, device, None, &conds((hp2, wp2))?, &[(0usize, a1.clone(), 1.0f32)], seed, &hooks).map_err(err)?.0,
            };
            drop(dit2);
            synaptix_core::tensor::ops::conv_filter_cache_gc();
            return Ok((l3, Some(a1), None, Some(l1)));
        }
        if dfr {
            // DFR (эталон LTX-2 `dfr_pipeline.py`, этапы 1–2 без временных раундов и эпилога).
            if !ltx_version_at_least(&ckpt.config.model_version, 2, 5) {
                return Err("DFR есть только у LTX-2.5".into());
            }
            struct Anc;
            impl Drop for Anc {
                fn drop(&mut self) {
                    synaptix_video_ltx23::runtime::set_ancestral(None);
                }
            }
            let _anc = Anc;
            let base = seed.unwrap_or(1);
            let (n_canvas, slots) = lp::dfr_canvas(out_frames);
            let fp_c = lp::fp_for_frames(n_canvas);
            eprintln!("[ltx] DFR: холст {n_canvas} кадров, слоты ключевых кадров {slots:?}, этап 1 {}×{}", wp1 * 32, hp1 * 32);
            synaptix_video_ltx23::runtime::set_ancestral(Some(base.wrapping_add(10_000)));
            // Условия: первый кадр или хвост прошлой сцены — чистые токены в начале.
            let conds_for = |g: (usize, usize), second: bool| -> R<(Vec<(usize, Tensor, f32)>, Vec<(usize, Tensor, f32)>)> {
                Ok(match &mode {
                    LtxMode::Frames { first: Some(p), .. } => {
                        let img = synaptix_io::image::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()))?;
                        (vec![(0usize, image_tokens(&img, g)?, 1.0f32)], Vec::new())
                    }
                    LtxMode::Continue { tail } => {
                        let v = if second { Some(&tail.video) } else { tail.video_stage1.as_ref() }
                            .filter(|t| t.dims()[3] == g.0 && t.dims()[4] == g.1)
                            .ok_or("DFR: у хвоста прошлой сцены нет латента нужного разрешения (нужен хвост от ролика с двумя стадиями или DFR того же размера)")?;
                        let ac = match &tail.audio {
                            Some(a) => vec![(0usize, a.clone(), 1.0f32)],
                            None => Vec::new(),
                        };
                        (vec![(0usize, latent_tokens(v)?.0, 1.0f32)], ac)
                    }
                    _ => (Vec::new(), Vec::new()),
                })
            };
            let (vc1, ac1) = conds_for((hp1, wp1), false)?;
            let (l1, a1, s1) = lp::denoise_dfr(&dit, &v_enc, &a_enc, fp_c, hp1, wp1, &lp::DISTILLED_SIGMAS, None, None, &slots, None, None, &vc1, &ac1, fps, device, seed, &hooks).map_err(err)?;
            drop(dit);
            release();
            let (l2, s2) = {
                let view = ckpt.view_on(device);
                let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
                let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
                let up = Upsampler::load(paths.upscaler.as_ref().unwrap(), &mean, &std, device).map_err(|e| format!("апскейлер: {e}"))?;
                let l2 = up.upsample(&l1).map_err(err)?.to_dtype(compute).map_err(err)?;
                let s2 = up.upsample(&s1).map_err(err)?.to_dtype(compute).map_err(err)?;
                (l2, s2)
            };
            synaptix_core::tensor::ops::conv_filter_cache_clear();
            release();
            // Этап 2: DiT с IC-LoRA детализации (сила 0.5), референс — латент этапа 1.
            synaptix_core::gen_status::phase("DFR: загрузка DiT с LoRA детализации");
            let dl = LoraWeights::open(paths.detail_lora.as_ref().unwrap(), device, 0.5).map_err(|e| format!("LoRA детализации: {e}"))?;
            let dckpt = ckpt.view_on(Device::Cpu).with_lora(std::sync::Arc::new(dl));
            let dit2 = AvDit::load_with(&dckpt, device, compute, quant_dit, offload).map_err(|e| format!("DiT: {e}"))?;
            synaptix_core::gen_status::phase("DFR: детализация");
            synaptix_video_ltx23::runtime::set_ancestral(Some(base.wrapping_add(20_000)));
            let (vc2, ac2) = conds_for((hp2, wp2), true)?;
            let (l3, _, _) = lp::denoise_dfr(&dit2, &v_enc, &a_enc, fp_c, hp2, wp2, &lp::STAGE2_SIGMAS, Some(&l2), Some(&a1), &slots, Some(&s2), Some((&l1, 2)), &vc2, &ac2, fps, device, seed, &hooks).map_err(err)?;
            drop(dit2);
            // Обрезка холста до запрошенной длины (видео и звук этапа 1).
            let l = l3.narrow(2, 0, fp.min(fp_c)).map_err(err)?.contiguous().map_err(err)?;
            let fa = lp::audio_token_count(fp, fps).min(a1.dims()[1]);
            let a = a1.narrow(1, 0, fa).map_err(err)?.contiguous().map_err(err)?;
            synaptix_core::tensor::ops::conv_filter_cache_gc();
            return Ok((l, Some(a), None, Some(l1.narrow(2, 0, fp.min(fp_c)).map_err(err)?.contiguous().map_err(err)?)));
        }
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
                LtxMode::Continue { tail } => {
                    let v_pref = if init.is_none() {
                        tail.video_stage1
                            .as_ref()
                            .filter(|s1| s1.dims()[3] == g.0 && s1.dims()[4] == g.1)
                            .unwrap_or(&tail.video)
                    } else {
                        &tail.video
                    };
                    let (v_tok, _, _, _) = latent_tokens(v_pref)?;
                    let a_conds: Vec<(usize, Tensor, f32)> = match &tail.audio {
                        Some(a) => vec![(0, a.clone(), 1.0)],
                        None => Vec::new(),
                    };
                    let (l, a) = lp::denoise_av_cond(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, fps, device, nag, &[(0, v_tok, 1.0)], &a_conds, seed, &hooks).map_err(err)?;
                    (l, a, None)
                }
                LtxMode::Extend { video, tail_frames } => {
                    // Кодируем хвост исходного ролика (последние tail_frames кадров + звук) в латент
                    // и замораживаем в начале нового ролика, как в Continue.
                    let dur_src = media_seconds(video).unwrap_or(5.0);
                    let tail_sec = (*tail_frames as f64 / fps).min(dur_src);
                    let start_sec = (dur_src - tail_sec).max(0.0);
                    let dir = scratch("ext_tail")?;
                    let ok = Command::new("ffmpeg")
                        .args(["-v", "error", "-y", "-ss", &format!("{start_sec:.3}"), "-i"])
                        .arg(video)
                        .args(["-vf", &format!("fps={fps},scale={}:{}:force_original_aspect_ratio=increase,crop={}:{}", g.1 * 32, g.0 * 32, g.1 * 32, g.0 * 32)])
                        .args(["-frames:v", &tail_frames.to_string()])
                        .arg(dir.join("f%05d.png"))
                        .status()
                        .map(|s| s.success())
                        .unwrap_or(false);
                    if !ok {
                        let _ = std::fs::remove_dir_all(&dir);
                        return Err(format!("{}: хвост видео не прочитан", video.display()));
                    }
                    let mut vframes = Vec::with_capacity(*tail_frames);
                    for i in 1..=*tail_frames {
                        let p = dir.join(format!("f{i:05}.png"));
                        if !p.exists() { break; }
                        let img = synaptix_io::image::load_image(&p, device).map_err(err)?;
                        vframes.push(img.contiguous().map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, g.0 * 32, g.1 * 32]).map_err(err)?);
                    }
                    let _ = std::fs::remove_dir_all(&dir);
                    if vframes.is_empty() {
                        return Err(format!("{}: хвост видео пуст", video.display()));
                    }
                    while vframes.len() < *tail_frames {
                        vframes.push(vframes.last().unwrap().clone());
                    }
                    let refs: Vec<&Tensor> = vframes.iter().collect();
                    let vid_t = Tensor::cat(&refs, 2).map_err(err)?.contiguous().map_err(err)?;
                    let v_lat = venc.as_ref().unwrap().encode(&vid_t).map_err(|e| format!("VAE: {e}"))?;
                    let (v_tok, t_ctx, _, _) = latent_tokens(&v_lat)?;
                    let mut a_conds: Vec<(usize, Tensor, f32)> = Vec::new();
                    if has_audio_stream(video) {
                        let fa_ctx = lp::audio_token_count(t_ctx, fps);
                        if let Ok(all_pcm) = ffmpeg_f32(video, 16000, 1) {
                            let need_s = ((tail_sec * 16000.0).round() as usize).min(all_pcm.len());
                            if need_s > 0 {
                                let tail_pcm = all_pcm[all_pcm.len() - need_s..].to_vec();
                                if let Ok(aenc) = AudioVaeEncoder::load(&ckpt, device) {
                                    if let Ok(mel) = ltx_log_mel(&[tail_pcm], device) {
                                        if let Ok(enc) = aenc.encode(&mel) {
                                            if let Ok(a_lat) = fit_audio_tokens(enc, fa_ctx) {
                                                a_conds.push((0, a_lat, 1.0));
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    let (l, a) = lp::denoise_av_cond(&dit, &v_enc, &a_enc, fp, g.0, g.1, sigmas, vi, ai, fps, device, nag, &[(0, v_tok, 1.0)], &a_conds, seed, &hooks).map_err(err)?;
                    (l, a, None)
                }
                LtxMode::Frames { first, last } => {
                    let load = |p: &PathBuf| synaptix_io::image::load_image(p, Device::Cpu).map_err(|e| format!("{}: {e}", p.display()));
                    let conds = match first {
                        Some(p) => vec![(0usize, image_tokens(&load(p)?, g)?, 1.0f32)],
                        None => Vec::new(),
                    };
                    match last {
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
                LtxMode::Sheet { image } => {
                    // Лист — статичное видео в выходном разрешении (масштаб референса 1),
                    // не короче ролика и не меньше 121 кадра (корзина обучения).
                    let img = synaptix_io::image::load_image(image, Device::Cpu).map_err(|e| format!("{}: {e}", image.display()))?;
                    let (ph, pw) = (g.0 * 32, g.1 * 32);
                    let one = synaptix_io::image::fit_image(&img, pw, ph, true).map_err(err)?.to_device(device).map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, ph, pw]).map_err(err)?;
                    let n = lp::out_frame_count(lp::fp_for_frames(out_frames.max(121)));
                    let copies: Vec<&Tensor> = std::iter::repeat_n(&one, n).collect();
                    let frames = Tensor::cat(&copies, 2).map_err(err)?.contiguous().map_err(err)?;
                    let lat = venc.as_ref().unwrap().encode(&frames).map_err(|e| format!("VAE: {e}"))?;
                    drop(frames);
                    let (tok, f_, h_, w_) = latent_tokens(&lat)?;
                    let pos = lp::ref_video_positions(f_, h_, w_, fps, 1);
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
        // LTX-2.5: Euler-ancestral на обоих этапах, шум — seed+10000 / seed+20000.
        let ltx25 = ltx_version_at_least(&ckpt.config.model_version, 2, 5) && std::env::var("SYN_LTX_SAMPLER").as_deref() != Ok("euler") && !sheet_dev;
        let base = seed.unwrap_or(1);
        struct AncReset;
        impl Drop for AncReset {
            fn drop(&mut self) {
                synaptix_video_ltx23::runtime::set_ancestral(None);
            }
        }
        let _anc = AncReset;
        synaptix_video_ltx23::runtime::set_ancestral(ltx25.then(|| base.wrapping_add(10_000)));
        // Лист на полном чекпойнте: `steps` > 8 — расписание LTX2Scheduler на столько шагов.
        let (mut l, mut a, mut wave) = stage((hp1, wp1), &lp::DISTILLED_SIGMAS, None)?;
        let stage1_saved = if two_stage { Some(l.clone()) } else { None };
        if two_stage {
            let view = ckpt.view_on(device);
            let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
            let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
            let up = Upsampler::load(paths.upscaler.as_ref().unwrap(), &mean, &std, device).map_err(|e| format!("апскейлер: {e}"))?;
            let l2 = up.upsample(&l).map_err(err)?.to_dtype(compute).map_err(err)?;
            drop(up);
            synaptix_core::tensor::ops::conv_filter_cache_clear();
            synaptix_video_ltx23::runtime::set_ancestral(ltx25.then(|| base.wrapping_add(20_000)));
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
        Ok((l, Some(a), wave, stage1_saved))
    })?;
    let (latent, a_tok, input_wave, stage1_saved) = (latent.0, latent.1, latent.2, latent.3);
    // Отладка: итоговый латент сырым f32 `[128, F, H, W]` (подбор коэффициентов превью).
    if let Some(p) = std::env::var_os("LTX_DUMP_LATENT") {
        let d = latent.dims().to_vec();
        let v = latent.to_dtype(DType::F32).map_err(err)?.to_device(Device::Cpu).map_err(err)?.flatten_all().map_err(err)?.to_vec1::<f32>().map_err(err)?;
        let path = PathBuf::from(p).with_extension(format!("{}x{}x{}.f32", d[2], d[3], d[4]));
        std::fs::write(&path, v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()).map_err(err)?;
    }
    let mut pending_tail: Option<crate::tail::TailLatent> = None;
    let tail_path = if let Some(tp) = &req.tail_out {
        let ltx_lat = crate::ltx_stages::LtxLatent {
            video: latent.clone(),
            audio: a_tok.as_ref().unwrap().clone(),
            fp: latent.dims()[2],
            hp: latent.dims()[3],
            wp: latent.dims()[4],
            fps,
            overlap_frames,
            stage1_video: stage1_saved,
            label: String::new(),
        };
        let model_name = req.model_id.as_deref().unwrap_or("ltx-2-3");
        // Сохраняется после декода: видео-латент хвоста перекодируется из кадров (см. ниже).
        pending_tail = Some(ltx_lat.tail(req.context_frames.unwrap_or(0), model_name)?);
        Some(tp.clone())
    } else {
        None
    };
    release();
    synaptix_core::gen_status::phase("декодер VAE: видео");
    // LTX-2.5: временной апскейлер латента ×2 (F → 2F−1 латентных кадров, вдвое чаще кадры).
    let mut fps_mul = 1.0;
    let latent = match (&paths.temporal_upscaler, &mode) {
        (Some(tp), LtxMode::Text | LtxMode::Frames { .. }) if req.fps_x2 => {
            synaptix_core::gen_status::phase("временной апскейлер ×2");
            let view = ckpt.view_on(device);
            let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
            let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
            let up = Upsampler::load(tp, &mean, &std, device).map_err(|e| format!("временной апскейлер: {e}"))?;
            if !up.is_temporal() {
                return Err("файл временного апскейлера — пространственный".into());
            }
            let l = synaptix_core::grad::no_grad(|| up.upsample(&latent)).map_err(err)?.to_dtype(compute).map_err(err)?;
            drop(up);
            synaptix_core::tensor::ops::conv_filter_cache_clear();
            release();
            fps_mul = 2.0;
            l
        }
        _ => latent,
    };
    let rgb = synaptix_core::grad::no_grad(|| -> R<Tensor> {
        if let (true, Some(dp)) = (req.diff_decoder, &paths.diff_vae) {
            let dec = synaptix_video_ltx23::diff_vae::DiffVaeDecoder::load(dp, device).map_err(|e| format!("диффузионный декодер: {e}"))?;
            return dec.decode(&latent, req.seed.wrapping_add(30_000)).map_err(|e| format!("диффузионный декодер: {e}"));
        }
        let vae = VaeDecoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?;
        vae.decode(&latent).map_err(|e| format!("VAE: {e}"))
    })?;
    release();
    // Звук: исходный (липсинк, LipDub) или сгенерированный.
    let pcm: Vec<f32> = match input_wave {
        Some(w) => w,
        None => {
            synaptix_core::gen_status::phase("декодер VAE: звук");
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
    let mut frames = lp::rgb_to_frames(&rgb).map_err(err)?;
    drop(rgb);
    // Хвост для продолжения — из готовых кадров, а не срезом латента: у
    // причинного VAE первый латентный кадр кодирует один кадр, остальные — по
    // восемь; срез середины ролика на местах 0…t давал мягкие первые кадры
    // продолжения и скачок резкости через ~16 кадров («переход» на стыке).
    // Эталон LTX-2 (chunks) так же переносит 8k+1 пиксельных кадров.
    if let (Some(mut tail), Some(tp)) = (pending_tail.take(), &req.tail_out) {
        if fps_mul == 1.0 && frames.len() >= tail.meta.overlap_frames {
            synaptix_core::gen_status::phase("хвост для продолжения");
            let last = &frames[frames.len() - tail.meta.overlap_frames..];
            let venc = VaeEncoder::load(&ckpt, device).map_err(|e| format!("VAE: {e}"))?;
            let encode = |gh: usize, gw: usize| -> R<Tensor> {
                let (ph, pw) = (gh * 32, gw * 32);
                let mut fr = Vec::with_capacity(last.len());
                for f in last {
                    fr.push(synaptix_io::image::fit_image(f, pw, ph, true).map_err(err)?.to_device(device).map_err(err)?.affine(2.0, -1.0).map_err(err)?.contiguous().map_err(err)?.reshape(vec![1, 3, 1, ph, pw]).map_err(err)?);
                }
                let refs: Vec<&Tensor> = fr.iter().collect();
                let x = Tensor::cat(&refs, 2).map_err(err)?.contiguous().map_err(err)?;
                let l = synaptix_core::grad::no_grad(|| venc.encode(&x)).map_err(|e| format!("VAE: {e}"))?;
                l.to_device(Device::Cpu).map_err(err)?.to_dtype(DType::F32).map_err(err)
            };
            let v = encode(hp, wp)?;
            if v.dims()[2] == tail.meta.context_frames {
                tail.video = v;
                if tail.video_stage1.is_some() {
                    let (h1, w1) = lp::stage1_grid(hp, wp);
                    tail.video_stage1 = Some(encode(h1, w1)?);
                }
            }
            drop(venc);
            release();
        }
        tail.save(tp)?;
    }
    let mut pcm = pcm;
    if overlap_frames > 0 && frames.len() > overlap_frames {
        frames.drain(0..overlap_frames);
        let cut_frames = ((overlap_frames as f64 / fps) * 48_000.0).round() as usize;
        let cut_samples = (cut_frames * 2).min(pcm.len());
        if cut_samples > 0 && cut_samples < pcm.len() {
            pcm.drain(0..cut_samples);
        }
    }
    let n = frames.len();
    let continued = matches!(mode, LtxMode::Continue { .. });
    let label_mode = match &mode {
        LtxMode::Text => "по описанию".to_string(),
        LtxMode::Frames { first: Some(_), last: Some(_) } => "между первым и последним кадром".to_string(),
        LtxMode::Frames { last: Some(_), .. } => "к последнему кадру".to_string(),
        LtxMode::Frames { .. } => "с первого кадра".to_string(),
        LtxMode::Continue { tail } => format!("с хвоста ({} лат.)", tail.meta.context_frames),
        LtxMode::Retake { .. } => "retake отрезка".to_string(),
        LtxMode::Extend { .. } => "продление по латенту".to_string(),
        LtxMode::A2v { image: Some(_), .. } => "по звуку с первого кадра (липсинк)".to_string(),
        LtxMode::A2v { .. } => "по звуку (липсинк)".to_string(),
        LtxMode::Ic { canny: true, .. } => "по контурам образца (IC-LoRA)".to_string(),
        LtxMode::Ic { .. } => "по движению образца (IC-LoRA)".to_string(),
        LtxMode::Sheet { .. } => "по листу-референсу (IC-LoRA Ingredients)".to_string(),
        LtxMode::Lipdub { .. } => "LipDub".to_string(),
    };
    match &mode {
        LtxMode::Extend { video, .. } => {
            // Новое продолжение — следом за исходным роликом (перекрытие уже вырезано выше).
            let dir = scratch("ext")?;
            let part = dir.join("part.mp4");
            write_mp4(&frames, Some((&pcm, 48_000, 2)), fps, &part)?;
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
        }
        _ => write_mp4(&frames, Some((&pcm, 48_000, 2)), fps * fps_mul, out)?,
    }
    let name = if ltx_version_at_least(&ckpt.config.model_version, 2, 5) { "LTX-2.5" } else { "LTX-2.3" };
    let label = format!("{name} {}×{}, {:.1} с, {label_mode}{}", wp * 32, hp * 32, n as f64 / (fps * fps_mul), if dfr { ", DFR" } else if two_stage { ", две стадии" } else { "" });
    Ok(VideoOutput { width: wp * 32, height: hp * 32, frames: n, fps, label, tail: tail_path, continued_from_latent: continued })
}

#[cfg(test)]
mod lora_tests {
    #[test]
    fn lora_steps_from_name() {
        use std::path::Path;
        assert_eq!(super::lora_steps(Path::new("minimax_h3_fl2v_turbo_8step_v1.1_768p_bf16.safetensors")), Some(8));
        assert_eq!(super::lora_steps(Path::new("minimax_h3_fl2v_turbo_4step_v1.3_768p_bf16.safetensors")), Some(4));
        assert_eq!(super::lora_steps(Path::new("style.safetensors")), None);
    }
}

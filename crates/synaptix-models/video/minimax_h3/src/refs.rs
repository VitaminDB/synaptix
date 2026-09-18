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

const IMAGE_EXT: &[&str] = &["png", "jpg", "jpeg", "webp", "bmp", "heic", "heif", "tif", "tiff"];
const VIDEO_EXT: &[&str] = &["mp4", "mov", "mkv", "webm", "m4v", "avi"];
const AUDIO_EXT: &[&str] = &["wav", "mp3", "flac", "ogg", "opus", "m4a", "aac"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefSource {
    Image(PathBuf),
    /// `use_audio` — брать ли дорожку видео как его собственный `<Audio j>`.
    Video { path: PathBuf, use_audio: bool },
    Audio(PathBuf),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    Image,
    Video,
    Audio,
}

impl RefKind {
    /// Тип референса по расширению файла.
    pub fn of_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        if IMAGE_EXT.contains(&ext.as_str()) {
            Some(Self::Image)
        } else if VIDEO_EXT.contains(&ext.as_str()) {
            Some(Self::Video)
        } else if AUDIO_EXT.contains(&ext.as_str()) {
            Some(Self::Audio)
        } else {
            None
        }
    }
}

impl RefSource {
    pub fn from_path(path: impl Into<PathBuf>, use_audio: bool) -> Result<Self, H3Error> {
        let path = path.into();
        match RefKind::of_path(&path) {
            Some(RefKind::Image) => Ok(Self::Image(path)),
            Some(RefKind::Video) => Ok(Self::Video { path, use_audio }),
            Some(RefKind::Audio) => Ok(Self::Audio(path)),
            None => Err(H3Error::Config(format!(
                "референс {}: по расширению не понять, картинка это, видео или аудио",
                path.display()
            ))),
        }
    }

    pub fn kind(&self) -> RefKind {
        match self {
            Self::Image(_) => RefKind::Image,
            Self::Video { .. } => RefKind::Video,
            Self::Audio(_) => RefKind::Audio,
        }
    }

    pub fn path(&self) -> &Path {
        match self {
            Self::Image(p) | Self::Audio(p) => p,
            Self::Video { path, .. } => path,
        }
    }
}

/// Размер картинки-референса.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RefImageSize {
    /// Как у выпущенной модели: короткая сторона 2048. Лучше держит
    /// идентичность, но строки референса едут через каждый шаг денойза —
    /// в разы медленнее.
    Max,
    /// Уменьшить (не увеличивая) до площади генерируемого кадра.
    #[default]
    Match,
}

#[derive(Debug, Clone, Copy)]
pub struct RefOptions {
    pub image_size: RefImageSize,
    pub target_width: usize,
    pub target_height: usize,
    /// Число кадров генерации (`17n + 5`): по нему обрезаются видео и звук.
    pub frame_count: usize,
}

/// Лимиты выпущенного чекпойнта: 9 картинок, 3 видео, 3 аудио, 12 всего;
/// аудио не бывает единственным референсом. Дорожка видео в лимит аудио не
/// входит — она часть своего видео.
pub fn validate(sources: &[RefSource]) -> Result<(), H3Error> {
    let count = |k: RefKind| sources.iter().filter(|s| s.kind() == k).count();
    for (kind, limit, name) in [
        (RefKind::Image, MAX_IMAGES, "картинок"),
        (RefKind::Video, MAX_VIDEOS, "видео"),
        (RefKind::Audio, MAX_AUDIOS, "аудио"),
    ] {
        if count(kind) > limit {
            return Err(H3Error::Config(format!(
                "референсы: {name} не больше {limit}, передано {}",
                count(kind)
            )));
        }
    }
    if sources.len() > MAX_REFS {
        return Err(H3Error::Config(format!(
            "референсы: не больше {MAX_REFS} всего, передано {}",
            sources.len()
        )));
    }
    if !sources.is_empty() && count(RefKind::Audio) == sources.len() {
        return Err(H3Error::Config(
            "референсы: аудио не может быть единственным — нужна хотя бы одна картинка или видео"
                .into(),
        ));
    }
    Ok(())
}

/// `round()` Питона: половина — к чётному. Эталон округляет им размеры холста.
pub fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 {
        r - x.signum()
    } else {
        r
    }
}

fn snap_multiple(v: f64) -> usize {
    let m = CANVAS_MULTIPLE as f64;
    (round_half_even(v / m) * m).max(m) as usize
}

fn check_aspect(width: usize, height: usize, what: &str) -> Result<(), H3Error> {
    if width == 0 || height == 0 {
        return Err(H3Error::Config(format!("{what}: пустой кадр {width}x{height}")));
    }
    if width > 4 * height || height > 4 * width {
        return Err(H3Error::Config(format!(
            "{what}: соотношение сторон {width}x{height} вне диапазона 1:4 … 4:1"
        )));
    }
    Ok(())
}

/// Холст H3 для данного аспекта: короткая сторона 768, площадь не больше
/// 768×1344, обе стороны кратны 32. Возвращает `(width, height)`.
pub fn resolve_canvas_size(aspect_w: usize, aspect_h: usize) -> Result<(usize, usize), H3Error> {
    check_aspect(aspect_w, aspect_h, "холст")?;
    let ratio = aspect_w as f64 / aspect_h as f64;
    let short = CANVAS_SHORT_EDGE as f64;
    let (mut w, mut h) = if ratio >= 1.0 { (short * ratio, short) } else { (short, short / ratio) };
    let area = w * h;
    if area > CANVAS_MAX_PIXELS as f64 {
        let scale = (CANVAS_MAX_PIXELS as f64 / area).sqrt();
        w *= scale;
        h *= scale;
    }
    Ok((snap_multiple(w), snap_multiple(h)))
}

/// Размер, в котором картинка-референс идёт в VAE и в башню. `(width, height)`.
pub fn ref_image_size(
    width: usize,
    height: usize,
    mode: RefImageSize,
    target_width: usize,
    target_height: usize,
) -> Result<(usize, usize), H3Error> {
    check_aspect(width, height, "картинка-референс")?;
    let scale = match mode {
        RefImageSize::Max => REF_IMAGE_SHORT_EDGE as f64 / width.min(height) as f64,
        RefImageSize::Match => {
            let area = (target_width * target_height) as f64 / (width * height) as f64;
            area.sqrt().min(1.0)
        }
    };
    Ok((snap_multiple(width as f64 * scale), snap_multiple(height as f64 * scale)))
}

/// Сколько кадров видео-референса уйдёт в VAE: вниз до `17n + 5`, чтобы
/// энкодер не добивал клип повтором кадра.
pub fn snap_ref_frames(frames: usize) -> usize {
    let n = frames.saturating_sub(FRAME_GRID_BASE) / FRAME_GRID_STEP;
    n.max(1) * FRAME_GRID_STEP + FRAME_GRID_BASE
}

/// Кадры, которые видит энкодер: каждый `24 / 2`-й из нормализованных.
pub fn encoder_sample_indices(frames: usize) -> Vec<usize> {
    let stride = FPS / ENCODER_SAMPLE_FPS;
    let mut out: Vec<usize> = Vec::new();
    let mut cursor = 0.0f64;
    while (round_half_even(cursor) as usize) < frames {
        let i = round_half_even(cursor) as usize;
        if out.last().is_none_or(|last| i > *last) {
            out.push(i);
        }
        cursor += stride;
    }
    out
}

/// Метка времени каждого vision-блока: среднее по группе из `tps` кадров,
/// последняя неполная группа добита своим последним кадром.
pub fn block_timestamps(sampled: usize, tps: usize) -> Vec<f32> {
    let tps = tps.max(1);
    let mut ts: Vec<f64> = (0..sampled).map(|i| i as f64 / ENCODER_SAMPLE_FPS).collect();
    while ts.len() % tps != 0 {
        ts.push(*ts.last().unwrap_or(&0.0));
    }
    ts.chunks(tps).map(|g| ((g[0] + g[tps - 1]) / 2.0) as f32).collect()
}

pub struct RefImage {
    pub rgb: Vec<u8>,
    pub width: usize,
    pub height: usize,
}

/// Стерео 32 кГц, планарно: сначала весь левый канал, потом правый.
pub struct RefWave {
    pub samples: Vec<f32>,
    pub len: usize,
}

pub struct RefVideo {
    /// `count` кадров RGB подряд, 24 fps, уже на холсте своего аспекта.
    pub frames: Vec<u8>,
    pub count: usize,
    pub width: usize,
    pub height: usize,
    pub audio: Option<RefWave>,
}

pub enum RefMedia {
    Image(RefImage),
    Video(RefVideo),
    Audio(RefWave),
}

impl RefMedia {
    pub fn describe(&self) -> String {
        match self {
            Self::Image(i) => format!("картинка {}x{}", i.width, i.height),
            Self::Video(v) => format!(
                "видео {}x{}, {} кадров{}",
                v.width,
                v.height,
                v.count,
                if v.audio.is_some() { " + звук" } else { "" }
            ),
            Self::Audio(a) => format!("аудио {:.1} с", a.len as f64 / AUDIO_SAMPLE_RATE as f64),
        }
    }
}

struct Probe {
    width: usize,
    height: usize,
    has_audio: bool,
}

fn run(cmd: &mut Command, what: &str) -> Result<Vec<u8>, H3Error> {
    let out = cmd.output().map_err(|e| H3Error::Io(format!("{what}: {e}")))?;
    if !out.status.success() {
        return Err(H3Error::Io(format!(
            "{what}: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        )));
    }
    Ok(out.stdout)
}

/// Размер кадра как его покажет плеер (с учётом поворота из display matrix —
/// ffmpeg при декодировании разворачивает кадр сам) и наличие аудиопотока.
fn probe(path: &Path) -> Result<Probe, H3Error> {
    if !path.is_file() {
        return Err(H3Error::Io(format!("референс не найден: {}", path.display())));
    }
    let raw = run(
        Command::new("ffprobe")
            .args(["-v", "error", "-show_entries"])
            .arg("stream=codec_type,width,height:stream_side_data=rotation")
            .args(["-of", "json"])
            .arg(path),
        "ffprobe",
    )?;
    let json: serde_json::Value =
        serde_json::from_slice(&raw).map_err(|e| H3Error::Io(format!("ffprobe: {e}")))?;
    let streams = json["streams"].as_array().cloned().unwrap_or_default();
    let mut probe = Probe { width: 0, height: 0, has_audio: false };
    for s in &streams {
        match s["codec_type"].as_str() {
            Some("audio") => probe.has_audio = true,
            Some("video") if probe.width == 0 => {
                let w = s["width"].as_u64().unwrap_or(0) as usize;
                let h = s["height"].as_u64().unwrap_or(0) as usize;
                let rotation = s["side_data_list"]
                    .as_array()
                    .and_then(|l| l.iter().find_map(|d| d["rotation"].as_f64()))
                    .unwrap_or(0.0);
                let quarter = (rotation / 90.0).round() as i64;
                (probe.width, probe.height) = if quarter % 2 != 0 { (h, w) } else { (w, h) };
            }
            _ => {}
        }
    }
    Ok(probe)
}

fn decode_rgb(
    path: &Path,
    width: usize,
    height: usize,
    video_frames: Option<usize>,
) -> Result<Vec<u8>, H3Error> {
    let scale = format!("scale={width}:{height}:flags=lanczos");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-i"]).arg(path).arg("-an");
    match video_frames {
        Some(n) => {
            cmd.args(["-vf", &format!("fps={FPS},{scale}"), "-frames:v", &n.to_string()]);
        }
        None => {
            cmd.args(["-vf", &scale, "-frames:v", "1"]);
        }
    }
    cmd.args(["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]);
    let bytes = run(&mut cmd, "ffmpeg")?;
    let frame = width * height * 3;
    if bytes.is_empty() || bytes.len() % frame != 0 {
        return Err(H3Error::Io(format!(
            "ffmpeg: {} — {} байт не кратно кадру {width}x{height}",
            path.display(),
            bytes.len()
        )));
    }
    Ok(bytes)
}

fn decode_wave(path: &Path, max_seconds: f64) -> Result<RefWave, H3Error> {
    let bytes = run(
        Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(path)
            .args(["-vn", "-t", &format!("{max_seconds:.6}")])
            .args(["-ac", "2", "-ar", &AUDIO_SAMPLE_RATE.to_string()])
            .args(["-f", "f32le", "-"]),
        "ffmpeg",
    )?;
    let len = bytes.len() / 8;
    if len == 0 {
        return Err(H3Error::Io(format!("в {} нет звука", path.display())));
    }
    let mut samples = vec![0f32; len * 2];
    for (i, pair) in bytes.chunks_exact(8).enumerate() {
        samples[i] = f32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]);
        samples[len + i] = f32::from_le_bytes([pair[4], pair[5], pair[6], pair[7]]);
    }
    Ok(RefWave { samples, len })
}

/// Декодирует и нормализует референсы в порядке списка. Ничего не держит на
/// карте: дальше этим питаются и энкодер, и VAE, в разное время.
pub fn decode(sources: &[RefSource], opts: &RefOptions) -> Result<Vec<RefMedia>, H3Error> {
    validate(sources)?;
    let seconds = opts.frame_count as f64 / FPS;
    let mut out = Vec::with_capacity(sources.len());
    for src in sources {
        let media = match src {
            RefSource::Image(path) => {
                let p = probe(path)?;
                let (w, h) = ref_image_size(
                    p.width,
                    p.height,
                    opts.image_size,
                    opts.target_width,
                    opts.target_height,
                )?;
                RefMedia::Image(RefImage { rgb: decode_rgb(path, w, h, None)?, width: w, height: h })
            }
            RefSource::Video { path, use_audio } => {
                let p = probe(path)?;
                let (w, h) = resolve_canvas_size(p.width, p.height)?;
                let frames = decode_rgb(path, w, h, Some(opts.frame_count))?;
                let count = frames.len() / (w * h * 3);
                let need = FRAME_GRID_STEP + FRAME_GRID_BASE;
                if count < need {
                    return Err(H3Error::Config(format!(
                        "видео-референс {}: {count} кадров при 24 fps, нужно не меньше {need} \
                         (модель ждёт 2–15 с)",
                        path.display()
                    )));
                }
                let audio = if *use_audio && p.has_audio {
                    Some(decode_wave(path, seconds)?)
                } else {
                    None
                };
                RefMedia::Video(RefVideo { frames, count, width: w, height: h, audio })
            }
            RefSource::Audio(path) => RefMedia::Audio(decode_wave(path, seconds)?),
        };
        out.push(media);
    }
    Ok(out)
}

fn rgb_to_chw01(rgb: &[u8], width: usize, height: usize) -> Result<Tensor, H3Error> {
    let plane = width * height;
    let mut chw = vec![0f32; 3 * plane];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        chw[i] = px[0] as f32 / 255.0;
        chw[plane + i] = px[1] as f32 / 255.0;
        chw[2 * plane + i] = px[2] as f32 / 255.0;
    }
    Ok(Tensor::from_vec(chw, vec![3, height, width], Device::Cpu)?)
}

/// `[1, 3, len, H, W]` в `[-1, 1]` — вход VAE.
fn rgb_to_clip(rgb: &[u8], len: usize, width: usize, height: usize) -> Result<Tensor, H3Error> {
    let plane = width * height;
    let mut out = vec![0f32; 3 * len * plane];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        let (t, p) = (i / plane, i % plane);
        for c in 0..3 {
            out[(c * len + t) * plane + p] = px[c] as f32 / 127.5 - 1.0;
        }
    }
    Ok(Tensor::from_vec(out, vec![1, 3, len, height, width], Device::Cpu)?)
}

/// Что энкодеру нужно от референсов: пункты презентации и vision-блоки — в том
/// порядке, в каком презентация встретит их pad-серии.
pub struct RefEncoderInputs {
    pub items: Vec<RefItem>,
    pub vision: Vec<(Tensor, ImageGrid)>,
}

pub fn encoder_inputs(
    encoder: &EncoderHandle,
    media: &[RefMedia],
) -> Result<RefEncoderInputs, H3Error> {
    let enc = encoder.encoder();
    let vis = |e: synaptix_vlm_qwen3::model::VisionError| H3Error::Load(e.to_string());
    let mut items = Vec::with_capacity(media.len());
    let mut vision = Vec::new();
    for m in media {
        match m {
            RefMedia::Image(img) => {
                let chw = rgb_to_chw01(&img.rgb, img.width, img.height)?;
                let (patches, grid) = enc.prepare_image_with(&chw, REF_IMAGE_LIMITS).map_err(vis)?;
                items.push(RefItem::Image { grid });
                vision.push((patches, grid));
            }
            RefMedia::Audio(_) => items.push(RefItem::Audio),
            RefMedia::Video(v) => {
                let frame = v.width * v.height * 3;
                let picked = encoder_sample_indices(v.count);
                let frames = picked
                    .iter()
                    .map(|i| rgb_to_chw01(&v.frames[i * frame..(i + 1) * frame], v.width, v.height))
                    .collect::<Result<Vec<_>, _>>()?;
                let groups =
                    enc.prepare_video_groups(&frames, VideoPixelLimits::default()).map_err(vis)?;
                let stamps = block_timestamps(picked.len(), enc.temporal_patch_size());
                if groups.len() != stamps.len() {
                    return Err(H3Error::Layout(format!(
                        "видео-референс: {} vision-блоков при {} метках времени",
                        groups.len(),
                        stamps.len()
                    )));
                }
                let blocks = groups
                    .iter()
                    .zip(&stamps)
                    .map(|((_, grid), seconds)| VideoBlock { grid: *grid, seconds: *seconds })
                    .collect();
                items.push(RefItem::Video { blocks, with_audio: v.audio.is_some() });
                vision.extend(groups);
            }
        }
    }
    Ok(RefEncoderInputs { items, vision })
}

/// Латентная сторона референсов: блоки раскладки и готовые cond-строки.
pub struct RefLatents {
    pub blocks: Vec<RefBlock>,
    pub cond_rows: CondRows,
}

fn encode_wave(vae: &AudioVae, wave: &RefWave, device: Device) -> Result<Tensor, H3Error> {
    let x = Tensor::from_vec(wave.samples.clone(), vec![1, 2, wave.len], Device::Cpu)?
        .to_device(device)?;
    vae.encode(&x)
}

/// Кодирует референсы: картинки и видео — видео-VAE, дорожки — аудио-VAE
/// (нужен `AudioVae::load_full`). Визуальные строки зашумляются до уровня
/// кондиционирования модели (`t = 0.999`), звук идёт чистым. Порядок строк —
/// порядок сегментов раскладки: у видео звук перед кадрами, но потоки
/// раздельные, так что внутри каждого потока это просто порядок списка.
pub fn encode_latents(
    media: &[RefMedia],
    vae: &VaeEncoder,
    audio_vae: Option<&AudioVae>,
    patch: [usize; 3],
    device: Device,
    seed: u64,
) -> Result<RefLatents, H3Error> {
    let need_audio = || {
        audio_vae.ok_or_else(|| H3Error::Load("для аудио-референса нужен энкодер audio VAE".into()))
    };
    let mut blocks = Vec::with_capacity(media.len());
    let mut video_rows: Vec<Tensor> = Vec::new();
    let mut audio_rows: Vec<Tensor> = Vec::new();

    let push_visual = |z: &Tensor, rows: &mut Vec<Tensor>| -> Result<(), H3Error> {
        let r = patchify_video(&z.to_dtype(DType::F32)?, patch)?;
        let draw = seed.wrapping_add(rows.len() as u64);
        rows.push(apply_noise_aug(&r, VISUAL_COND_TIMESTEP, draw)?);
        Ok(())
    };

    for m in media {
        match m {
            RefMedia::Image(img) => {
                let x = rgb_to_clip(&img.rgb, 1, img.width, img.height)?;
                let z = vae.encode(&x)?;
                push_visual(&z, &mut video_rows)?;
                blocks.push(RefBlock::Image {
                    latent_h: img.height / VAE_SPATIAL_RATIO,
                    latent_w: img.width / VAE_SPATIAL_RATIO,
                });
            }
            RefMedia::Audio(wave) => {
                let z = encode_wave(need_audio()?, wave, device)?;
                blocks.push(RefBlock::Audio { latent_t: z.dims()[3] });
                audio_rows.push(pack_audio(&z.to_dtype(DType::F32)?)?);
            }
            RefMedia::Video(v) => {
                let mut audio_latent_t = 0;
                if let Some(wave) = &v.audio {
                    let z = encode_wave(need_audio()?, wave, device)?;
                    audio_latent_t = z.dims()[3];
                    audio_rows.push(pack_audio(&z.to_dtype(DType::F32)?)?);
                }
                let frame = v.width * v.height * 3;
                let total = snap_ref_frames(v.count).min(v.count);
                let z = vae.encode_clips(total, &mut |start, len| {
                    rgb_to_clip(&v.frames[start * frame..(start + len) * frame], len, v.width, v.height)
                })?;
                push_visual(&z, &mut video_rows)?;
                blocks.push(RefBlock::Video {
                    latent_t: z.dims()[2],
                    latent_h: v.height / VAE_SPATIAL_RATIO,
                    latent_w: v.width / VAE_SPATIAL_RATIO,
                    audio_latent_t,
                });
            }
        }
    }

    let cat = |rows: &[Tensor]| -> Result<Option<Tensor>, H3Error> {
        if rows.is_empty() {
            return Ok(None);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Ok(Some(Tensor::cat(&refs, 0)?))
    };
    Ok(RefLatents {
        blocks,
        cond_rows: CondRows { video: cat(&video_rows)?, audio: cat(&audio_rows)? },
    })
}

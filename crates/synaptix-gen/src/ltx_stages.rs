//! LTX-2.3 по стадиям — для нодовых движков (граф службы synaptixd), с теми
//! же возможностями, что у прежних родных нод Synthos: текст (Gemma →
//! коннекторы), NAG, картинка-условие (кадр 0 — замена, дальше — ключевой
//! кадр), Stage1 на половинной сетке → апскейл ×2 → Stage2, retake отрезка,
//! IC-LoRA по образцу (как есть, контуры Canny, глубина Depth Anything),
//! LipDub, видео по звуку; декодеры видео и звука.
//!
//! Тяжёлое держит модель: DiT (22B, общий для Stage1 и Stage2 — квантовать
//! заново дорого) и Gemma (общая для описания и NAG). Держатель отпускает их
//! по [`LtxModel::release`] — перед декодом, по простою и когда видеокарта
//! нужна другим моделям. [`crate::video::ltx`] — отдельный путь «одним
//! вызовом» (свои умолчания: кадр-условие с обрезкой, NAG по умолчанию).

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_gemma3::pipeline::GemmaPipeline;
use synaptix_video_ltx23::dit::{dit_resident_bytes, AvDit};
use synaptix_video_ltx23::loader::LoraWeights;
use synaptix_video_ltx23::pipeline as lp;
use synaptix_video_ltx23::LtxCheckpoint;

use crate::video::LtxPaths;

type R<T> = Result<T, String>;

fn err(e: impl std::fmt::Display) -> String {
    e.to_string()
}

const COMPUTE: DType = DType::BF16;
/// Длина контекста Gemma у LTX: перцивер-коннектор обучен на S=1024.
const GEMMA_CTX: usize = 1024;
/// Частота вокодера LTX-2.3.
pub const AUDIO_RATE: u32 = 48_000;

struct DitBox {
    dit: AvDit,
    _pin: Option<synaptix_core::device::cuda::OffloadPinCacheGuard>,
    _ckpt: Arc<LtxCheckpoint>,
}

#[derive(Default)]
struct Held {
    dit: Option<Arc<DitBox>>,
    gemma: Option<Arc<GemmaPipeline>>,
    used: Option<Instant>,
}

/// Открытая модель: пути, точность; DiT и Gemma — по требованию.
pub struct LtxModel {
    pub paths: LtxPaths,
    device: Device,
    quant_dit: DType,
    quant_enc: DType,
    lora: Option<(PathBuf, f32)>,
    depth: Option<PathBuf>,
    ckpt: Mutex<Option<Arc<LtxCheckpoint>>>,
    held: Mutex<Held>,
}

/// Закодированное описание: контексты видео и звука.
pub struct LtxCond {
    pub video: Tensor,
    pub audio: Tensor,
}

/// NAG: закодированный «негатив» и его параметры.
pub struct LtxNag {
    pub enc: Tensor,
    pub scale: f32,
    pub alpha: f32,
    pub tau: f32,
}

/// Картинка-условие: кадр 0 — замена первого кадра, дальше — ключевой кадр.
pub struct LtxImage {
    pub image: Tensor,
    pub strength: f32,
    pub frame: usize,
}

/// Латенты ролика: видео (сетка fp×hp×wp) и звуковые токены.
pub struct LtxLatent {
    pub video: Tensor,
    pub audio: Tensor,
    pub fp: usize,
    pub hp: usize,
    pub wp: usize,
    pub fps: f64,
    pub label: String,
}

impl LtxLatent {
    pub fn width(&self) -> usize {
        self.wp * 32
    }
    pub fn height(&self) -> usize {
        self.hp * 32
    }
    pub fn seconds(&self) -> f64 {
        lp::out_frame_count(self.fp) as f64 / self.fps
    }
}

/// Размер и длительность ролика.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    pub width: usize,
    pub height: usize,
    pub seconds: f64,
    pub fps: f64,
}

impl Shape {
    fn grid(&self) -> (usize, usize) {
        lp::latent_grid(if self.width > 0 { self.width } else { 960 }, if self.height > 0 { self.height } else { 544 })
    }
    fn fp(&self) -> usize {
        lp::frames_for_duration(if self.seconds > 0.0 { self.seconds } else { 5.0 }, self.fps)
    }
}

/// Управляющий сигнал IC-LoRA из образца.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Control {
    /// Образец как есть.
    None,
    /// Контуры Canny (пороги 0..1).
    Canny { low: f32, high: f32 },
    /// Карта глубины Depth Anything V2.
    Depth,
}

impl LtxImage {
    /// Картинка-условие с диска.
    pub fn load(path: &Path, strength: f32, frame: usize) -> R<LtxImage> {
        let image = synaptix_io::image::load_image(path, Device::Cpu).map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(LtxImage { image, strength, frame })
    }
}

/// Ход и отмена денойза.
pub struct Hooks<'a> {
    pub cancel: &'a AtomicBool,
    pub progress: &'a (dyn Fn(usize, usize) + Sync),
}

fn trim(dev: Device) {
    if let Device::Cuda(o) = dev {
        let _ = synaptix_core::device::cuda::synchronize_all(o);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(o);
    }
}

fn ltx_err(e: synaptix_video_ltx23::LtxError) -> String {
    match e {
        synaptix_video_ltx23::LtxError::Cancelled => "прогон остановлен".into(),
        other => other.to_string(),
    }
}

/// Открыть модель. `quant` — DiT (`nvfp4`, `mxfp8`, `bf16`; пусто — формат
/// квантованного бандла, у плотного файла — плотный);
/// энкодер Gemma — NVFP4 при NVFP4-DiT, иначе MXFP8. `lora` — свой адаптер
/// поверх DiT (IC-LoRA, distilled), `depth` — каталог Depth Anything V2.
pub fn open(paths: LtxPaths, device: Device, quant: Option<&str>, lora: Option<(PathBuf, f32)>, depth: Option<PathBuf>) -> R<LtxModel> {
    if !device.is_cuda() {
        return Err("LTX-2.3 работает только на видеокарте".into());
    }
    if !paths.model.exists() {
        return Err(format!("модели нет: {}", paths.model.display()));
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let quant_dit = crate::video::quant_setting(quant).or_else(|| crate::video::stored_format(&paths.model)).unwrap_or(COMPUTE);
    let quant_enc = if quant_dit == DType::NVFP4 { DType::NVFP4 } else { DType::MXFP8 };
    Ok(LtxModel { paths, device, quant_dit, quant_enc, lora: lora.filter(|(_, s)| *s > 0.0), depth, ckpt: Mutex::new(None), held: Mutex::new(Held::default()) })
}

impl LtxModel {
    pub fn is_quantized(&self) -> bool {
        self.quant_dit.is_quantized()
    }

    fn ckpt(&self) -> R<Arc<LtxCheckpoint>> {
        let mut g = self.ckpt.lock().map_err(err)?;
        if let Some(c) = g.as_ref() {
            return Ok(c.clone());
        }
        let c = Arc::new(LtxCheckpoint::open(&self.paths.model, Device::Cpu, DType::BF16).map_err(|e| format!("LTX {}: {e}", self.paths.model.display()))?);
        *g = Some(c.clone());
        Ok(c)
    }

    /// Отпустить DiT и Gemma (память видеокарты — драйверу).
    pub fn release(&self) {
        if let Ok(mut h) = self.held.lock() {
            if h.dit.is_none() && h.gemma.is_none() {
                return;
            }
            *h = Held::default();
        }
        trim(self.device);
    }

    /// Отпустить, если не трогали дольше `idle`. `true` — отпустили.
    pub fn release_idle(&self, idle: Duration) -> bool {
        let stale = self.held.lock().map(|h| (h.dit.is_some() || h.gemma.is_some()) && h.used.is_some_and(|t| t.elapsed() > idle)).unwrap_or(false);
        if stale {
            self.release();
        }
        stale
    }

    pub fn holds(&self) -> bool {
        self.held.lock().map(|h| h.dit.is_some() || h.gemma.is_some()).unwrap_or(false)
    }

    fn gemma(&self) -> R<Arc<GemmaPipeline>> {
        if let Some(g) = self.held.lock().map_err(err)?.gemma.clone() {
            return Ok(g);
        }
        // DiT и Gemma вместе на карту не помещаются.
        if let Ok(mut h) = self.held.lock() {
            h.dit = None;
        }
        trim(self.device);
        let prec = PrecisionConfig { compute: COMPUTE, attn_w: self.quant_enc, mlp_w: self.quant_enc, lm_head: DType::BF16, embed: DType::BF16, kv: DType::BF16 };
        let g = Arc::new(GemmaPipeline::load_with_precision(&self.paths.gemma, self.device, prec, Some(GEMMA_CTX)).map_err(|e| format!("Gemma {}: {e}", self.paths.gemma.display()))?);
        let mut h = self.held.lock().map_err(err)?;
        h.gemma = Some(g.clone());
        h.used = Some(Instant::now());
        Ok(g)
    }

    /// DiT (общий для стадий). `tv_max` — токенов целевой сетки (решает,
    /// держать DiT на карте целиком или стримить блоки с хоста).
    fn dit(&self, tv_max: usize) -> R<Arc<DitBox>> {
        {
            let mut h = self.held.lock().map_err(err)?;
            h.used = Some(Instant::now());
            if let Some(d) = h.dit.clone() {
                return Ok(d);
            }
            h.gemma = None;
        }
        let dev = self.device;
        trim(dev);
        let ckpt = self.ckpt()?;
        let offload = self.quant_dit == COMPUTE || {
            let dit_b = dit_resident_bytes(&ckpt, self.quant_dit, COMPUTE);
            let vae_b: usize = ckpt.infos().filter(|(n, _, _)| n.starts_with("vae.")).map(|(_, _, s)| COMPUTE.bytes_for_numel(s.iter().product())).sum();
            let act_b = tv_max * 4096 * 2 * 28;
            let pad_b = (tv_max + 256) * (16384 * 2 + 16384 * 2 + 12288 * 2);
            let need = dit_b + vae_b + act_b + pad_b + (1usize << 30);
            let ord = if let Device::Cuda(o) = dev { o } else { 0 };
            synaptix_core::device::cuda::mem_info(ord).map(|(free, _)| need > free).unwrap_or(true)
        };
        let pin = (offload && self.quant_dit == COMPUTE).then(|| {
            let g = synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused(&ckpt.shard_bytes());
            g.resume();
            g
        });
        let dit = match &self.lora {
            Some((p, s)) => {
                let lw = LoraWeights::open(p, dev, *s).map_err(|e| format!("LoRA {}: {e}", p.display()))?;
                let ck = ckpt.view_on(dev).with_lora(Arc::new(lw));
                AvDit::load_with(&ck, dev, COMPUTE, self.quant_dit, offload).map_err(|e| format!("DiT (+LoRA): {e}"))?
            }
            None => AvDit::load_with(&ckpt, dev, COMPUTE, self.quant_dit, offload).map_err(|e| format!("DiT: {e}"))?,
        };
        let d = Arc::new(DitBox { dit, _pin: pin, _ckpt: ckpt });
        let mut h = self.held.lock().map_err(err)?;
        h.dit = Some(d.clone());
        h.used = Some(Instant::now());
        Ok(d)
    }

    fn encode_gemma(&self, prompt: &str) -> R<(Vec<Tensor>, Vec<u32>)> {
        let gemma = self.gemma()?;
        gemma.encode_for_ltx(prompt, GEMMA_CTX, self.device).map_err(|e| format!("Gemma: {e}"))
    }

    /// Описание → контексты видео и звука (Gemma → коннекторы).
    pub fn encode(&self, prompt: &str) -> R<LtxCond> {
        use synaptix_video_ltx23::text_encoder::{AudioTextConditioner, VideoTextConditioner};
        let dev = self.device;
        synaptix_core::device::cuda::set_offload_pinned(true);
        let r = (|| {
            let (states, mask) = self.encode_gemma(prompt)?;
            let ckpt = self.ckpt()?;
            let view = ckpt.view_on(dev);
            // Видео-коннектор отпускается до загрузки звукового — на малой карте оба не влезают.
            let video = VideoTextConditioner::load(&view, dev, COMPUTE).map_err(err)?.forward(&states, &mask).map_err(err)?;
            trim(dev);
            let audio = AudioTextConditioner::load(&view, dev, COMPUTE).map_err(err)?.forward(&states, &mask).map_err(err)?;
            Ok(LtxCond { video, audio })
        })();
        synaptix_core::device::cuda::set_offload_pinned(false);
        trim(dev);
        r
    }

    /// NAG: «негатив» → видео-контекст с параметрами (пусто — умолчание LTX).
    pub fn nag(&self, prompt: &str, scale: f32, alpha: f32, tau: f32) -> R<LtxNag> {
        use synaptix_video_ltx23::text_encoder::VideoTextConditioner;
        let dev = self.device;
        let prompt = if prompt.trim().is_empty() { lp::DEFAULT_NAG_PROMPT } else { prompt };
        synaptix_core::device::cuda::set_offload_pinned(true);
        let r = (|| {
            let (states, mask) = self.encode_gemma(prompt)?;
            trim(dev);
            let ckpt = self.ckpt()?;
            let enc = VideoTextConditioner::load(&ckpt.view_on(dev), dev, COMPUTE).map_err(err)?.forward(&states, &mask).map_err(err)?;
            Ok(LtxNag { enc, scale, alpha, tau })
        })();
        synaptix_core::device::cuda::set_offload_pinned(false);
        trim(dev);
        r
    }

    fn vae_encoder(&self) -> R<synaptix_video_ltx23::vae::VaeEncoder> {
        synaptix_video_ltx23::vae::VaeEncoder::load(&*self.ckpt()?, self.device).map_err(|e| format!("VAE: {e}"))
    }

    /// Картинка → токены условия на сетке (hp, wp): растянуть, VAE, патчи.
    fn image_tokens(&self, image: &Tensor, hp: usize, wp: usize) -> R<Tensor> {
        let (ph, pw) = (hp * 32, wp * 32);
        let img = synaptix_io::image::resize_bilinear(image, ph, pw)
            .map_err(err)?
            .contiguous()
            .and_then(|t| t.affine(2.0, -1.0))
            .and_then(|t| t.contiguous())
            .and_then(|t| t.reshape(vec![1, 3, 1, ph, pw]))
            .map_err(err)?;
        let lat = self.vae_encoder()?.encode(&img).map_err(|e| format!("VAE: {e}"))?;
        lp::frame_latent_to_tokens(&lat).map_err(err)
    }

    /// Кадры ролика `[1,3,F,H,W]` в [−1, 1] (не больше `n`). VAE всего ролика
    /// рядом с DiT на карту не помещается — DiT отпускается заранее.
    fn video_frames(&self, path: &Path, ph: usize, pw: usize, n: usize) -> R<Tensor> {
        self.release();
        let dir = std::env::temp_dir().join(format!("synaptix_ltx_vin_{}_{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
        std::fs::create_dir_all(&dir).map_err(err)?;
        let r = (|| {
            let ok = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-i"])
                .arg(path)
                .args(["-vf", &format!("scale={pw}:{ph}"), "-frames:v", &n.to_string()])
                .arg(dir.join("f%05d.png"))
                .status()
                .map(|s| s.success())
                .unwrap_or(false);
            if !ok {
                return Err(format!("{}: ffmpeg не прочитал кадры", path.display()));
            }
            let mut frames = Vec::with_capacity(n);
            for i in 1..=n {
                let p = dir.join(format!("f{i:05}.png"));
                if !p.exists() {
                    break;
                }
                let img = synaptix_io::image::load_image(&p, self.device).map_err(err)?;
                frames.push(img.contiguous().and_then(|t| t.affine(2.0, -1.0)).and_then(|t| t.contiguous()).and_then(|t| t.reshape(vec![1, 3, 1, ph, pw])).map_err(err)?);
            }
            if frames.is_empty() {
                return Err(format!("{}: нет кадров", path.display()));
            }
            let refs: Vec<&Tensor> = frames.iter().collect();
            Tensor::cat(&refs, 2).and_then(|t| t.contiguous()).map_err(err)
        })();
        let _ = std::fs::remove_dir_all(&dir);
        r
    }

    /// Кадры → токены образца `[1, F·H·W, 128]` и его латентная сетка.
    fn latent_tokens(&self, frames: &Tensor) -> R<(Tensor, usize, usize, usize)> {
        let lat = self.vae_encoder()?.encode(frames).map_err(|e| format!("VAE: {e}"))?;
        let d = lat.dims().to_vec();
        let (f, h, w) = (d[2], d[3], d[4]);
        Ok((lat.reshape(vec![1, 128, f * h * w]).and_then(|t| t.transpose(1, 2)).and_then(|t| t.contiguous()).map_err(err)?, f, h, w))
    }

    /// Звук → токены audio-VAE `[1, T, 128]`.
    fn audio_tokens(&self, path: &Path) -> R<Tensor> {
        use synaptix_video_ltx23::audio_vae::{ltx_log_mel, AudioVaeEncoder};
        let pcm = crate::video::ffmpeg_f32(path, 16000, 1)?;
        let mel = ltx_log_mel(&[pcm], self.device).map_err(err)?;
        AudioVaeEncoder::load(&*self.ckpt()?, self.device).map_err(|e| format!("VAE звука: {e}"))?.encode(&mel).map_err(err)
    }

    fn hooks<'a>(&self, h: &'a Hooks<'a>, on_step: &'a (dyn Fn(lp::DenoiseProgress) + Sync)) -> lp::DenoiseHooks<'a> {
        let _ = h;
        lp::DenoiseHooks { progress: Some(on_step), cancel: Some(h.cancel) }
    }

    /// Stage1: денойз на половинной сетке (с NAG и картинкой-условием).
    pub fn stage1(&self, cond: &LtxCond, nag: Option<&LtxNag>, image: Option<&LtxImage>, shape: Shape, seed: u64, h: &Hooks) -> R<LtxLatent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.quant_dit.is_quantized());
        let dev = self.device;
        let fps = shape.fps;
        let (hp, wp) = shape.grid();
        let fp = shape.fp();
        let (hp1, wp1) = lp::stage1_grid(hp, wp);
        let toks = image.map(|i| self.image_tokens(&i.image, hp1, wp1)).transpose()?;
        let dit = self.dit(fp * hp * wp)?;
        let on_step = |p: lp::DenoiseProgress| (h.progress)(p.step, p.total);
        let hooks = self.hooks(h, &on_step);
        let v_nag = nag.map(|n| (&n.enc, n.scale, n.alpha, n.tau));
        let (v, a) = match (image, &toks) {
            (Some(i), Some(t)) if i.frame > 0 => {
                let pos = lp::keyframe_positions(hp1, wp1, i.frame, fps);
                lp::denoise_av_append(&dit.dit, &cond.video, &cond.audio, fp, hp1, wp1, &lp::DISTILLED_SIGMAS, None, None, false, Some((t, &pos, i.strength)), None, fps, dev, Some(seed), &hooks).map_err(ltx_err)?
            }
            (Some(i), Some(t)) => lp::denoise_av(&dit.dit, &cond.video, &cond.audio, fp, hp1, wp1, &lp::DISTILLED_SIGMAS, None, None, fps, dev, v_nag, &[(0, t.clone(), i.strength)], Some(seed), &hooks).map_err(ltx_err)?,
            _ => lp::denoise_av(&dit.dit, &cond.video, &cond.audio, fp, hp1, wp1, &lp::DISTILLED_SIGMAS, None, None, fps, dev, v_nag, &[], Some(seed), &hooks).map_err(ltx_err)?,
        };
        Ok(LtxLatent { video: v, audio: a, fp, hp: hp1, wp: wp1, fps, label: format!("LTX-2.3 Stage1 {}×{}", wp1 * 32, hp1 * 32) })
    }

    /// Апскейл латента ×2 (звук — как был).
    pub fn upscale(&self, lat: &LtxLatent) -> R<LtxLatent> {
        let path = self.paths.upscaler.as_ref().ok_or("у модели нет апскейлера LTX-2.3 ×2")?;
        let dev = self.device;
        let ckpt = self.ckpt()?;
        let view = ckpt.view_on(dev);
        let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
        let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
        let up = synaptix_video_ltx23::upscaler::Upsampler::load(path, &mean, &std, dev).map_err(|e| format!("апскейлер: {e}"))?;
        let v = up.upsample(&lat.video).map_err(err)?.to_dtype(COMPUTE).map_err(err)?;
        drop(up);
        synaptix_core::tensor::ops::conv_filter_cache_clear();
        Ok(LtxLatent { video: v, audio: lat.audio.clone(), fp: lat.fp, hp: lat.hp * 2, wp: lat.wp * 2, fps: lat.fps, label: format!("{} → ×2", lat.label) })
    }

    /// Stage2: доводка латента (после апскейла) с тем же условием.
    pub fn stage2(&self, lat: &LtxLatent, cond: &LtxCond, image: Option<&LtxImage>, seed: u64, h: &Hooks) -> R<LtxLatent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.quant_dit.is_quantized());
        let dev = self.device;
        let toks = image.map(|i| self.image_tokens(&i.image, lat.hp, lat.wp)).transpose()?;
        let dit = self.dit(lat.fp * lat.hp * lat.wp)?;
        let on_step = |p: lp::DenoiseProgress| (h.progress)(p.step, p.total);
        let hooks = self.hooks(h, &on_step);
        let (v, a) = match (image, &toks) {
            (Some(i), Some(t)) if i.frame > 0 => {
                let pos = lp::keyframe_positions(lat.hp, lat.wp, i.frame, lat.fps);
                lp::denoise_av_append(&dit.dit, &cond.video, &cond.audio, lat.fp, lat.hp, lat.wp, &lp::STAGE2_SIGMAS, Some(&lat.video), Some(&lat.audio), false, Some((t, &pos, i.strength)), None, lat.fps, dev, Some(seed), &hooks).map_err(ltx_err)?
            }
            (Some(i), Some(t)) => lp::denoise_av(&dit.dit, &cond.video, &cond.audio, lat.fp, lat.hp, lat.wp, &lp::STAGE2_SIGMAS, Some(&lat.video), Some(&lat.audio), lat.fps, dev, None, &[(0, t.clone(), i.strength)], Some(seed), &hooks).map_err(ltx_err)?,
            _ => lp::denoise_av(&dit.dit, &cond.video, &cond.audio, lat.fp, lat.hp, lat.wp, &lp::STAGE2_SIGMAS, Some(&lat.video), Some(&lat.audio), lat.fps, dev, None, &[], Some(seed), &hooks).map_err(ltx_err)?,
        };
        Ok(LtxLatent { video: v, audio: a, fp: lat.fp, hp: lat.hp, wp: lat.wp, fps: lat.fps, label: format!("LTX-2.3 {}×{}", lat.wp * 32, lat.hp * 32) })
    }

    /// Retake: перегенерировать отрезок `start..end` (с) исходного ролика.
    pub fn retake(&self, cond: &LtxCond, video: &Path, shape: Shape, start: f64, end: f64, seed: u64, h: &Hooks) -> R<LtxLatent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.quant_dit.is_quantized());
        let dev = self.device;
        let fps = shape.fps;
        let (hp_t, wp_t) = shape.grid();
        let out_frames = lp::out_frame_count(shape.fp());
        let frames = self.video_frames(video, hp_t * 32, wp_t * 32, out_frames)?;
        let source = self.vae_encoder()?.encode(&frames).map_err(|e| format!("VAE: {e}"))?;
        drop(frames);
        // Сетка — по настоящему латенту (временное сжатие VAE).
        let d = source.dims().to_vec();
        let (fp, hp, wp) = (d[2], d[3], d[4]);
        let dit = self.dit(fp * hp * wp)?;
        let on_step = |p: lp::DenoiseProgress| (h.progress)(p.step, p.total);
        let hooks = self.hooks(h, &on_step);
        let (v, a) = lp::denoise_av_retake(&dit.dit, &cond.video, &cond.audio, fp, hp, wp, &lp::DISTILLED_SIGMAS, &source, start, end, fps, dev, Some(seed), &hooks).map_err(ltx_err)?;
        Ok(LtxLatent { video: v, audio: a, fp, hp, wp, fps, label: format!("LTX-2.3 retake {start:.1}–{end:.1} с") })
    }

    /// IC-LoRA: ролик по образцу (масштаб `downscale`, сила `strength`,
    /// управляющий сигнал `control`). Адаптер — LoRA модели.
    #[allow(clippy::too_many_arguments)]
    pub fn ic_lora(&self, cond: &LtxCond, video: &Path, shape: Shape, downscale: usize, strength: f32, control: Control, seed: u64, h: &Hooks) -> R<LtxLatent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.quant_dit.is_quantized());
        let dev = self.device;
        let fps = shape.fps;
        let (hp, wp) = shape.grid();
        let fp = shape.fp();
        let ds = downscale.max(1);
        let mut frames = self.video_frames(video, hp * 32 / ds, wp * 32 / ds, lp::out_frame_count(fp))?;
        frames = match control {
            Control::None => frames,
            Control::Canny { low, high } => per_frame(&frames, |fr| synaptix_io::image::canny_rgb(fr, low, high).map_err(err))?,
            Control::Depth => {
                let dir = self.depth.as_ref().ok_or("для управления глубиной нужен каталог Depth Anything V2")?;
                let m = synaptix_depth_anything::DepthAnything::load(dir, dev).map_err(|e| format!("Depth Anything: {e}"))?;
                per_frame(&frames, |fr| m.depth_rgb(fr).map_err(err))?
            }
        };
        let (tok, f_, h_, w_) = self.latent_tokens(&frames)?;
        drop(frames);
        let pos = lp::ref_video_positions(f_, h_, w_, fps, ds);
        let dit = self.dit(fp * hp * wp)?;
        let on_step = |p: lp::DenoiseProgress| (h.progress)(p.step, p.total);
        let hooks = self.hooks(h, &on_step);
        let (v, a) = lp::denoise_av_append(&dit.dit, &cond.video, &cond.audio, fp, hp, wp, &lp::DISTILLED_SIGMAS, None, None, false, Some((&tok, &pos, strength)), None, fps, dev, Some(seed), &hooks).map_err(ltx_err)?;
        let what = match control {
            Control::None => "по образцу",
            Control::Canny { .. } => "по контурам образца",
            Control::Depth => "по глубине образца",
        };
        Ok(LtxLatent { video: v, audio: a, fp, hp, wp, fps, label: format!("LTX-2.3 IC-LoRA {what}") })
    }

    /// LipDub: исходный ролик переозвучен речью `audio` (две стадии сразу).
    pub fn lipdub(&self, cond: &LtxCond, video: &Path, audio: &Path, shape: Shape) -> R<LtxLatent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.quant_dit.is_quantized());
        let dev = self.device;
        let fps = shape.fps;
        let (hp, wp) = shape.grid();
        let fp = shape.fp();
        let out_frames = lp::out_frame_count(fp);
        let (hp1, wp1) = lp::stage1_grid(hp, wp);
        let refs = |g: (usize, usize)| -> R<(Tensor, Vec<f64>)> {
            let frames = self.video_frames(video, g.0 * 32, g.1 * 32, out_frames)?;
            let (tok, f_, h_, w_) = self.latent_tokens(&frames)?;
            Ok((tok, lp::pixel_coords(f_, h_, w_, fps)))
        };
        let (r1, p1) = refs((hp1, wp1))?;
        let (r2, p2) = refs((hp1 * 2, wp1 * 2))?;
        let aref = self.audio_tokens(audio)?;
        let dit = self.dit(fp * hp * wp)?;
        let ckpt = self.ckpt()?;
        let view = ckpt.view_on(dev);
        let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(err)?;
        let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(err)?;
        let up = synaptix_video_ltx23::upscaler::Upsampler::load(self.paths.upscaler.as_ref().ok_or("у модели нет апскейлера LTX-2.3 ×2")?, &mean, &std, dev).map_err(|e| format!("апскейлер: {e}"))?;
        let (v, a) = lp::generate_lipdub_latents(&dit.dit, &up, &cond.video, &cond.audio, &r1, &p1, &r2, &p2, &aref, fp, hp1, wp1, fps, dev).map_err(ltx_err)?;
        synaptix_core::tensor::ops::conv_filter_cache_clear();
        Ok(LtxLatent { video: v, audio: a, fp, hp: hp1 * 2, wp: wp1 * 2, fps, label: "LTX-2.3 LipDub".into() })
    }

    /// Видео по звуку: звук заморожен на обеих стадиях (Stage1 → ×2 → Stage2).
    pub fn audio_to_video(&self, cond: &LtxCond, audio: &Path, shape: Shape, seed: u64, h: &Hooks) -> R<LtxLatent> {
        let _fp8 = synaptix_core::precision::AttentionFp8Guard::new(self.quant_dit.is_quantized());
        let dev = self.device;
        let fps = shape.fps;
        let (hp, wp) = shape.grid();
        let fp = shape.fp();
        let (hp1, wp1) = lp::stage1_grid(hp, wp);
        let fa = lp::audio_token_count(fp, fps);
        let audio_lat = crate::video::fit_audio_tokens(self.audio_tokens(audio)?, fa)?;
        let dit = self.dit(fp * hp * wp)?;
        let total = 2 * (lp::DISTILLED_SIGMAS.len() - 1);
        let done = std::sync::atomic::AtomicUsize::new(0);
        let on_step = |_: lp::DenoiseProgress| {
            let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            (h.progress)(n.min(total), total);
        };
        let hooks = self.hooks(h, &on_step);
        let (l1, _) = lp::denoise_av_append(&dit.dit, &cond.video, &cond.audio, fp, hp1, wp1, &lp::DISTILLED_SIGMAS, None, Some(&audio_lat), true, None, None, fps, dev, Some(seed), &hooks).map_err(ltx_err)?;
        let up = self.upscale(&LtxLatent { video: l1, audio: audio_lat.clone(), fp, hp: hp1, wp: wp1, fps, label: String::new() })?;
        let (v, a) = lp::denoise_av_append(&dit.dit, &cond.video, &cond.audio, fp, hp1 * 2, wp1 * 2, &lp::STAGE2_SIGMAS, Some(&up.video), Some(&audio_lat), true, None, None, fps, dev, Some(seed), &hooks).map_err(ltx_err)?;
        Ok(LtxLatent { video: v, audio: a, fp, hp: hp1 * 2, wp: wp1 * 2, fps, label: "LTX-2.3 по звуку".into() })
    }

    /// Латент видео → кадры `[3,H,W]` 0..1. DiT отпускается (VAE нужна память).
    pub fn decode_video(&self, lat: &LtxLatent) -> R<Vec<Tensor>> {
        self.release();
        let rgb = synaptix_core::grad::no_grad(|| -> R<Tensor> {
            let vae = synaptix_video_ltx23::vae::VaeDecoder::load(&*self.ckpt()?, self.device).map_err(|e| format!("VAE: {e}"))?;
            vae.decode(&lat.video).map_err(|e| format!("VAE: {e}"))
        })?;
        trim(self.device);
        lp::rgb_to_frames(&rgb).map_err(err)
    }

    /// Звуковые токены → стерео (перемежённые отсчёты, [`AUDIO_RATE`]).
    pub fn decode_audio(&self, lat: &LtxLatent) -> R<Vec<f32>> {
        use synaptix_video_ltx23::audio_vae::AudioVaeDecoder;
        use synaptix_video_ltx23::vocoder::VocoderWithBwe;
        let wave = synaptix_core::grad::no_grad(|| -> R<Tensor> {
            let avae = AudioVaeDecoder::load(&*self.ckpt()?, self.device).map_err(|e| format!("VAE звука: {e}"))?;
            let voc = VocoderWithBwe::load(&self.paths.model, self.device).map_err(|e| format!("вокодер: {e}"))?;
            lp::decode_audio_tokens(&avae, &voc, &lat.audio).map_err(err)
        })?;
        trim(self.device);
        let (ch, l) = (wave.dims()[1], wave.dims()[2]);
        let planar: Vec<f32> = wave.reshape(vec![ch * l]).map_err(err)?.to_dtype(DType::F32).map_err(err)?.to_device(Device::Cpu).map_err(err)?.to_vec1::<f32>().map_err(err)?;
        Ok((0..l).flat_map(|i| (0..ch).map(move |c| (c, i))).map(|(c, i)| planar[c * l + i]).collect())
    }
}

/// Обработать каждый кадр `[1,3,F,H,W]` ([−1, 1]) функцией над `[3,H,W]` (0..1).
fn per_frame(frames: &Tensor, f: impl Fn(&Tensor) -> R<Tensor>) -> R<Tensor> {
    let (n, h, w) = (frames.dims()[2], frames.dims()[3], frames.dims()[4]);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let fr = frames.narrow(2, i, 1).and_then(|t| t.contiguous()).and_then(|t| t.reshape(vec![3, h, w])).and_then(|t| t.affine(0.5, 0.5)).map_err(err)?;
        out.push(f(&fr)?.affine(2.0, -1.0).and_then(|t| t.reshape(vec![1, 3, 1, h, w])).map_err(err)?);
    }
    let refs: Vec<&Tensor> = out.iter().collect();
    Tensor::cat(&refs, 2).and_then(|t| t.contiguous()).map_err(err)
}

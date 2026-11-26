use std::sync::atomic::{AtomicBool, Ordering};

use synaptix_core::{device::Device, dtype::DType, error::SynaptixError, tensor::Tensor};

use crate::adaln::{mod_segments, AdalnCache, AdalnPlan, ModSegment};
use crate::config::{
    audio_latent_frames, frames_for_duration, latent_frames, latent_grid, snap_frame_count,
    AUDIO_COND_TIMESTEP, VISUAL_COND_TIMESTEP,
};
use crate::dit::H3Dit;
use crate::guider::{apply_cfg, GuiderParams};
use crate::layout::{Keyframe, LayoutRequest, PackedLayout, RefBlock, SegmentKind};
use crate::loader::H3Checkpoint;
use crate::rope::RopeTables;
use crate::runtime;
use crate::scheduler::H3Scheduler;
use crate::H3Error;

fn default_sampler() -> SamplerKind {
    static KIND: std::sync::OnceLock<SamplerKind> = std::sync::OnceLock::new();
    *KIND.get_or_init(|| {
        if matches!(std::env::var("H3_SAMPLER").as_deref(), Ok("euler")) {
            SamplerKind::Euler
        } else {
            SamplerKind::ResMultistep
        }
    })
}

pub fn dump_step() -> usize {
    static STEP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *STEP.get_or_init(|| {
        std::env::var("H3_DUMP_STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    })
}

pub fn dump_text(name: &str, body: &str) {
    let Ok(dir) = std::env::var("H3_DUMP_DIR") else { return };
    let p = std::path::Path::new(&dir);
    let _ = std::fs::create_dir_all(p);
    let _ = std::fs::write(p.join(format!("{name}.json")), body);
}

pub fn dump_tensor(name: &str, t: &Tensor) {
    let Ok(dir) = std::env::var("H3_DUMP_DIR") else { return };
    let Ok(v) = t
        .to_dtype(DType::F32)
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
    else {
        return;
    };
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in &v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let p = std::path::Path::new(&dir);
    let _ = std::fs::create_dir_all(p);
    let _ = std::fs::write(p.join(format!("{name}.f32")), &bytes);
    let _ = std::fs::write(p.join(format!("{name}.shape")), format!("{:?}", t.dims()));
}

pub fn tensor_stats(name: &str, t: &Tensor) -> String {
    let v = match t.to_dtype(DType::F32).and_then(|x| x.flatten_all()).and_then(|x| x.to_vec1::<f32>()) {
        Ok(v) => v,
        Err(e) => return format!("{name}: <{e}>"),
    };
    if v.is_empty() {
        return format!("{name}: пусто");
    }
    let n = v.len() as f64;
    let mean = v.iter().map(|x| *x as f64).sum::<f64>() / n;
    let var = v.iter().map(|x| (*x as f64 - mean).powi(2)).sum::<f64>() / n;
    let nan = v.iter().filter(|x| !x.is_finite()).count();
    let (lo, hi) = v.iter().fold((f32::MAX, f32::MIN), |(l, h), x| (l.min(*x), h.max(*x)));
    format!(
        "{name}[{:?}] μ {mean:+.4} σ {:.4} [{lo:+.3}, {hi:+.3}]{}",
        t.dims(),
        var.sqrt(),
        if nan > 0 { format!(" NaN/Inf {nan}") } else { String::new() }
    )
}

type R<T> = Result<T, SynaptixError>;

#[derive(Debug, Clone, Copy)]
pub struct DenoiseProgress {
    pub step: usize,
    pub total: usize,
    pub sigma: f64,
}

#[derive(Default)]
pub struct DenoiseHooks<'a> {
    pub progress: Option<&'a (dyn Fn(DenoiseProgress) + Sync)>,
    pub cancel: Option<&'a AtomicBool>,
}

impl DenoiseHooks<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.map(|c| c.load(Ordering::Relaxed)).unwrap_or(false)
    }

    fn report(&self, step: usize, total: usize, sigma: f64) {
        if let Some(p) = self.progress {
            p(DenoiseProgress { step, total, sigma });
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    pub frame_count: usize,
    pub latent_t: usize,
    pub latent_h: usize,
    pub latent_w: usize,
    pub audio_t: usize,
}

impl Geometry {
    pub fn new(width: usize, height: usize, frame_count: usize) -> Self {
        let frames = snap_frame_count(frame_count);
        let (lh, lw) = latent_grid(width, height);
        Self {
            width,
            height,
            frame_count: frames,
            latent_t: latent_frames(frames),
            latent_h: lh,
            latent_w: lw,
            audio_t: audio_latent_frames(frames),
        }
    }

    pub fn from_duration(width: usize, height: usize, seconds: f64) -> Self {
        Self::new(width, height, frames_for_duration(seconds))
    }

    pub fn video_tokens(&self, patch: [usize; 3]) -> usize {
        (self.latent_t / patch[0]) * (self.latent_h / patch[1]) * (self.latent_w / patch[2])
    }
}

pub fn patchify_video(latent: &Tensor, patch: [usize; 3]) -> R<Tensor> {
    let d = latent.dims().to_vec();
    let (b, c, tf, hf, wf) = (d[0], d[1], d[2], d[3], d[4]);
    let (pt, ph, pw) = (patch[0], patch[1], patch[2]);
    let (t, h, w) = (tf / pt, hf / ph, wf / pw);
    latent
        .reshape(vec![b, c, t, pt, h, ph, w, pw])?
        .permute([0, 2, 4, 6, 1, 3, 5, 7])?
        .contiguous()?
        .reshape(vec![b * t * h * w, c * pt * ph * pw])
}

pub fn unpatchify_video(
    rows: &Tensor,
    t: usize,
    h: usize,
    w: usize,
    c: usize,
    patch: [usize; 3],
) -> R<Tensor> {
    let (pt, ph, pw) = (patch[0], patch[1], patch[2]);
    rows.reshape(vec![1, t, h, w, c, pt, ph, pw])?
        .permute([0, 4, 1, 5, 2, 6, 3, 7])?
        .contiguous()?
        .reshape(vec![1, c, t * pt, h * ph, w * pw])
}

pub fn pack_audio(latent: &Tensor) -> R<Tensor> {
    let d = latent.dims().to_vec();
    let (c, ch, t) = (d[1], d[2], d[3]);
    latent
        .reshape(vec![c, ch, t])?
        .permute([1, 2, 0])?
        .contiguous()?
        .reshape(vec![ch * t, c])
}

pub fn unpack_audio(rows: &Tensor, ch: usize) -> R<Tensor> {
    let d = rows.dims().to_vec();
    let t = d[0] / ch;
    let c = d[1];
    rows.reshape(vec![ch, t, c])?
        .permute([2, 0, 1])?
        .contiguous()?
        .reshape(vec![1, c, ch, t])
}

pub struct Conditioning {
    pub context: Tensor,
    pub text_tags: Vec<u8>,
}

pub struct CondRows {
    pub video: Option<Tensor>,
    pub audio: Option<Tensor>,
}

impl Default for CondRows {
    fn default() -> Self {
        Self { video: None, audio: None }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SamplerKind {
    ResMultistep,
    Euler,
}

pub struct DenoiseRequest<'a> {
    pub geometry: Geometry,
    pub cond: &'a Conditioning,
    pub negative: Option<&'a Conditioning>,
    pub keyframes: Vec<Keyframe>,
    pub refs: Vec<RefBlock>,
    pub cond_rows: CondRows,
    pub guider: GuiderParams,
    pub sampler: SamplerKind,
    pub seed: Option<u64>,
    pub init_video: Option<Tensor>,
    pub init_audio: Option<Tensor>,
    pub visual_cond_aug: Option<f32>,
    pub audio_cond_aug: Option<f32>,
    /// Хвост видео-латента предыдущего ролика `[1, 24, T_ctx, H, W]`: ставится
    /// в начало целевого видео и замораживается с ролью `VisualCond`.
    pub prefix_video: Option<Tensor>,
    /// Хвост аудио-латента предыдущего ролика `[1, 32, 2, A_ctx]`: ставится в
    /// начало целевого аудио и замораживается с ролью `AudioCond`.
    pub prefix_audio: Option<Tensor>,
}

impl<'a> DenoiseRequest<'a> {
    pub fn new(geometry: Geometry, cond: &'a Conditioning) -> Self {
        Self {
            geometry,
            cond,
            negative: None,
            keyframes: Vec::new(),
            refs: Vec::new(),
            cond_rows: CondRows::default(),
            guider: GuiderParams::positive_only(),
            sampler: default_sampler(),
            seed: None,
            init_video: None,
            init_audio: None,
            visual_cond_aug: None,
            audio_cond_aug: None,
            prefix_video: None,
            prefix_audio: None,
        }
    }
}

pub struct DenoiseOutput {
    pub video_latent: Tensor,
    pub audio_latent: Tensor,
}

pub struct PreparedRun {
    pub layout: PackedLayout,
    pub plan: AdalnPlan,
    pub segments: Vec<ModSegment>,
    pub rope: RopeTables,
    pub video_seg: (usize, usize, usize),
    pub audio_seg: (usize, usize, usize),
    pub refined_cond: Tensor,
    pub refined_negative: Option<Tensor>,
}

pub fn prepare(
    dit: &H3Dit,
    req: &DenoiseRequest<'_>,
    sched: &H3Scheduler,
) -> Result<PreparedRun, H3Error> {
    let g = req.geometry;
    dump_tensor("cond_hidden", &req.cond.context);
    dump_text("cond_tags", &format!("{:?}", req.cond.text_tags));
    let text_len = req.cond.context.dims()[1];
    let prefix_vt = req.prefix_video.as_ref().map(|t| t.dims().get(2).copied().unwrap_or(0)).unwrap_or(0);
    let prefix_at = req.prefix_audio.as_ref().map(|t| t.dims().get(3).copied().unwrap_or(0)).unwrap_or(0);
    let layout = PackedLayout::build(
        &LayoutRequest::new(text_len, g.latent_t, g.latent_h, g.latent_w, g.audio_t)
            .with_frame_count(g.frame_count)
            .with_keyframes(req.keyframes.clone())
            .with_refs(req.refs.clone())
            .with_prefix(prefix_vt, prefix_at),
    )?;
    dump_text(
        "positions",
        &format!(
            "[{}]",
            layout
                .positions
                .iter()
                .map(|p| format!("[{},{},{}]", p[0], p[1], p[2]))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    let adaln_sched = match speed_plan(req, sched.steps()) {
        Some((coarse_geo, coarse_steps)) => speed_stage2_sched(sched, coarse_geo, g, coarse_steps),
        None => sched.clone(),
    };
    let plan = AdalnPlan::build(&layout, &adaln_sched, req.visual_cond_aug, req.audio_cond_aug);
    let segments = mod_segments(&layout, &plan.roles, Some(&req.cond.text_tags));
    let rope = dit.rope_tables(&layout.positions)?;

    let vseg = layout
        .segment(SegmentKind::Video)
        .ok_or_else(|| H3Error::Layout("нет video-сегмента".into()))?;
    let aseg = layout
        .segment(SegmentKind::Audio)
        .ok_or_else(|| H3Error::Layout("нет audio-сегмента".into()))?;
    let vrow = plan.roles.index(plan.roles.role_for(SegmentKind::Video)) * crate::config::ADALN_MODALITIES;
    let arow = plan.roles.index(plan.roles.role_for(SegmentKind::Audio)) * crate::config::ADALN_MODALITIES;

    let refined_cond = dit.refine_text(&req.cond.context)?;
    dump_tensor("refined_cond", &refined_cond);
    let refined_negative = match &req.negative {
        Some(n) => Some(fit_text_rows(&dit.refine_text(&n.context)?, text_len)?),
        None => None,
    };

    Ok(PreparedRun {
        layout,
        plan,
        segments,
        rope,
        video_seg: (vseg.start, vseg.stop, vrow / crate::config::ADALN_MODALITIES),
        audio_seg: (aseg.start, aseg.stop, arow / crate::config::ADALN_MODALITIES),
        refined_cond,
        refined_negative,
    })
}

fn fit_text_rows(t: &Tensor, rows: usize) -> R<Tensor> {
    let have = t.dims()[0];
    if have == rows {
        return Ok(t.clone());
    }
    if have > rows {
        return t.narrow(0, 0, rows)?.contiguous();
    }
    let pad = Tensor::zeros(vec![rows - have, t.dims()[1]], t.dtype(), t.device())?;
    Tensor::cat(&[t, &pad], 0)
}

fn assemble_hidden(
    prep: &PreparedRun,
    refined: &Tensor,
    video_rows: &Tensor,
    audio_rows: &Tensor,
) -> R<Tensor> {
    let mut parts: Vec<Tensor> = Vec::with_capacity(prep.layout.segments.len());
    let mut voff = 0usize;
    let mut aoff = 0usize;
    for seg in &prep.layout.segments {
        let n = seg.len();
        if n == 0 {
            continue;
        }
        match seg.kind {
            SegmentKind::Text => parts.push(refined.narrow(0, 0, n)?.contiguous()?),
            SegmentKind::Cond | SegmentKind::RefImg | SegmentKind::Video => {
                parts.push(video_rows.narrow(0, voff, n)?.contiguous()?);
                voff += n;
            }
            SegmentKind::RefAudio | SegmentKind::Audio => {
                parts.push(audio_rows.narrow(0, aoff, n)?.contiguous()?);
                aoff += n;
            }
        }
    }
    if runtime::h3_prof() {
        eprintln!(
            "[h3-cat] {} · {} · {}",
            tensor_stats("refined", refined),
            tensor_stats("video_rows", video_rows),
            tensor_stats("audio_rows", audio_rows)
        );
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0)
}

/// Раскладка резервирует cond-строки по геометрии референсов, а сами строки
/// приходят из VAE отдельно — расхождение иначе всплывёт как выход за границы
/// посреди `merge_stream_rows`.
fn check_cond_rows(what: &str, cond: Option<&Tensor>, update: &[bool]) -> Result<(), H3Error> {
    let reserved = update.iter().filter(|u| !**u).count();
    let have = cond.map(|c| c.dims()[0]).unwrap_or(0);
    if reserved != have {
        return Err(H3Error::Layout(format!(
            "{what}: раскладка ждёт {reserved} cond-строк, закодировано {have} —              ключевые кадры/референсы запроса и их латенты не совпадают"
        )));
    }
    Ok(())
}

fn merge_stream_rows(
    target: &Tensor,
    cond: Option<&Tensor>,
    update: &[bool],
) -> R<Tensor> {
    let Some(cond) = cond else {
        return Ok(target.clone());
    };
    let mut parts: Vec<Tensor> = Vec::new();
    let mut t_off = 0usize;
    let mut c_off = 0usize;
    let mut i = 0usize;
    while i < update.len() {
        let flag = update[i];
        let mut j = i;
        while j < update.len() && update[j] == flag {
            j += 1;
        }
        let n = j - i;
        if flag {
            parts.push(target.narrow(0, t_off, n)?.contiguous()?);
            t_off += n;
        } else {
            parts.push(cond.narrow(0, c_off, n)?.contiguous()?);
            c_off += n;
        }
        i = j;
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0)
}

pub fn build_adaln_cache(
    dit: &H3Dit,
    ckpt: &H3Checkpoint,
    prep: &PreparedRun,
    cache_dtype: DType,
) -> Result<AdalnCache, H3Error> {
    dit.build_adaln_cache(&prep.plan, ckpt, cache_dtype)
}

pub fn init_latents(
    geometry: Geometry,
    latents_dim: usize,
    audio_dim: usize,
    device: Device,
    dtype: DType,
    seed: Option<u64>,
) -> Result<(Tensor, Tensor), H3Error> {
    let mut rng = synaptix_ops::rng::Philox4x32::new(seed.unwrap_or(0));
    let v_shape = vec![1, latents_dim, geometry.latent_t, geometry.latent_h, geometry.latent_w];
    let a_shape = vec![1, audio_dim, 2, geometry.audio_t];
    let v = randn(&v_shape, device, dtype, &mut rng)?;
    let a = randn(&a_shape, device, dtype, &mut rng)?;
    Ok((v, a))
}

fn randn(
    shape: &[usize],
    device: Device,
    dtype: DType,
    rng: &mut synaptix_ops::rng::Philox4x32,
) -> Result<Tensor, H3Error> {
    let n: usize = shape.iter().product();
    let mut v = vec![0f32; n];
    synaptix_ops::rng::fill_normal_f32(rng, &mut v);
    Ok(Tensor::from_vec(v, shape.to_vec(), device)?.to_dtype(dtype)?)
}

pub fn denoise_one(
    dit: &H3Dit,
    cache: &AdalnCache,
    prep: &PreparedRun,
    req: &DenoiseRequest<'_>,
    _sched: &H3Scheduler,
    step: usize,
) -> Result<(Tensor, Tensor), H3Error> {
    let cfg = &dit.cfg;
    let g = req.geometry;
    let patch = cfg.patch_size;
    let v_lat = req.init_video.clone().ok_or_else(|| H3Error::Layout("нет init_video".into()))?;
    let a_lat = req.init_audio.clone().ok_or_else(|| H3Error::Layout("нет init_audio".into()))?;

    let v_rows_t = patchify_video(&v_lat, patch)?;
    let a_rows_t = pack_audio(&a_lat)?;
    let cast = |t: &Option<Tensor>, like: &Tensor| -> R<Option<Tensor>> {
        t.as_ref().map(|c| c.to_device(like.device())?.to_dtype(like.dtype())).transpose()
    };
    let cond_video = cast(&req.cond_rows.video, &v_rows_t)?;
    let cond_audio = cast(&req.cond_rows.audio, &a_rows_t)?;
    let v_rows = merge_stream_rows(&v_rows_t, cond_video.as_ref(), &prep.layout.img_update)?;
    let a_rows = merge_stream_rows(&a_rows_t, cond_audio.as_ref(), &prep.layout.audio_update)?;
    let v_emb = dit.embed_video(&v_rows)?;
    let a_emb = dit.embed_audio(&a_rows)?;
    let hidden = assemble_hidden(prep, &prep.refined_cond, &v_emb, &a_emb)?;
    let (v_out, a_out) = dit.forward(
        &hidden,
        cache,
        step,
        &prep.segments,
        &prep.rope,
        prep.video_seg,
        prep.audio_seg,
    )?;
    let v_vel = unpatchify_video(
        &v_out.mul_scalar(-1.0)?,
        g.latent_t / patch[0],
        g.latent_h / patch[1],
        g.latent_w / patch[2],
        cfg.latents_dim,
        patch,
    )?;
    let a_vel = unpack_audio(&a_out.mul_scalar(-1.0)?, 2)?;
    Ok((v_vel, a_vel))
}

/// Настройки блочно-разреженного FP8-внимания (Sol-стиль) для шага денойза:
/// - `SYN_H3_SPARSE_ATTN=0` — выключить (по умолчанию включено при `seq_len ≥ 8192`);
/// - `SYN_H3_SPARSE_DENSE_STEPS` — число первых плотных шагов (по умолчанию `ceil(25%)`, т.е. 2 из 8, 5 из 20);
/// - `SYN_H3_SPARSE_KEEP` — доля дальних видео-блоков по top-k (по умолчанию `0.40`).
/// Префикс `[0..video_start)` (Text/Cond/Refs/Audio) и локальное окно ±1 кадр всегда плотные.
fn sparse_cfg_for_step(
    step: usize,
    total_steps: usize,
    prep: &PreparedRun,
) -> Option<synaptix_core::precision::AttentionSparseConfig> {
    if matches!(std::env::var("SYN_H3_SPARSE_ATTN").as_deref(), Ok("0") | Ok("false") | Ok("off")) {
        return None;
    }
    if prep.layout.seq_len < 8192 || total_steps < 2 {
        return None;
    }
    let dense_steps = std::env::var("SYN_H3_SPARSE_DENSE_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or_else(|| ((total_steps as f32 * 0.25).ceil() as usize).max(1));
    if step < dense_steps {
        return None;
    }
    let keep_ratio = std::env::var("SYN_H3_SPARSE_KEEP")
        .ok()
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.40)
        .clamp(0.15, 1.0);
    if keep_ratio >= 0.999 {
        return None;
    }
    Some(synaptix_core::precision::AttentionSparseConfig {
        protected_prefix: prep.video_seg.0,
        q_offset: 0,
        keep_ratio,
        frame_tokens: prep.layout.frame_rows,
    })
}

/// Параметры 2-стадийного coarse-to-fine семплинга (SPEED):
/// - `SYN_H3_SPEED=0` — выключить (по умолчанию включено для `steps ≥ 4` и
///   кадра выше порога сетки);
/// - `SYN_H3_SPEED_STEPS=N` — число грубых шагов (по умолчанию `round(35%)`,
///   т.е. 1/4, 3/8, 7/20).
/// Порог грубой сетки — минимум `34×60` латентов (`544×960`, кратно 32):
/// ниже этого порога на 768p H3 теряет кадрирование (крупный план уходит в общий).
fn speed_plan(req: &DenoiseRequest<'_>, steps: usize) -> Option<(Geometry, usize)> {
    if matches!(std::env::var("SYN_H3_SPEED").as_deref(), Ok("0") | Ok("false") | Ok("off")) {
        return None;
    }
    if !req.refs.is_empty() || req.init_video.is_some() {
        return None;
    }
    let force_steps = std::env::var("SYN_H3_SPEED_STEPS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok());
    let coarse_steps = match force_steps {
        Some(0) => return None,
        Some(n) => n.min(steps.saturating_sub(1)),
        None => {
            if steps < 4 {
                return None;
            }
            ((steps as f64 * 0.35).round() as usize).clamp(1, steps / 2)
        }
    };
    if coarse_steps == 0 || coarse_steps >= steps {
        return None;
    }
    let g = req.geometry;
    let cw_ch = speed_coarse_dims(g.width, g.height)?;
    Some((Geometry::new(cw_ch.0, cw_ch.1, g.frame_count), coarse_steps))
}

/// Вычислить грубое разрешение `(w, h)` с сохранением пропорций, кратностью 32
/// и порогом латентной сетки `≥ 34×60` (`544×960` px). `None`, если выигрыш < 30%.
fn speed_coarse_dims(width: usize, height: usize) -> Option<(usize, usize)> {
    let short = width.min(height) as f64;
    let long = width.max(height) as f64;
    let scale = 0.5f64.max(544.0 / short).max(960.0 / long);
    if scale >= 0.85 {
        return None;
    }
    let snap32 = |v: f64| ((v / 32.0).round().max(1.0) as usize) * 32;
    let cw = snap32(width as f64 * scale).min(width);
    let ch = snap32(height as f64 * scale).min(height);
    if cw * ch * 10 > width * height * 7 {
        return None;
    }
    Some((cw, ch))
}

/// Ортонормированный базис 1D DCT-II размера `n × n` (row-major).
fn dct2_basis(n: usize) -> Vec<f32> {
    let mut b = vec![0f32; n * n];
    let s0 = (1.0 / n as f64).sqrt();
    let s1 = (2.0 / n as f64).sqrt();
    let scale = std::f64::consts::PI / n as f64;
    for k in 0..n {
        let c = if k == 0 { s0 } else { s1 };
        let kf = k as f64;
        for j in 0..n {
            b[k * n + j] = (c * (scale * kf * (j as f64 + 0.5)).cos()) as f32;
        }
    }
    b
}

/// 2D ортонормированное DCT-II над плоскостями `[planes, H, W]`: `B_H · X · B_Wᵀ`.
fn dct2_planes(src: &[f32], planes: usize, h: usize, w: usize) -> Vec<f32> {
    let bh = dct2_basis(h);
    let bw = dct2_basis(w);
    let hw = h * w;
    let mut dst = vec![0f32; planes * hw];
    let mut tmp = vec![0f32; hw];
    for p in 0..planes {
        let sp = &src[p * hw..(p + 1) * hw];
        let dp = &mut dst[p * hw..(p + 1) * hw];
        for i in 0..h {
            let bi = &bh[i * h..(i + 1) * h];
            let tr = &mut tmp[i * w..(i + 1) * w];
            tr.fill(0.0);
            for k in 0..h {
                let bval = bi[k];
                let sr = &sp[k * w..(k + 1) * w];
                for j in 0..w {
                    tr[j] += bval * sr[j];
                }
            }
        }
        for i in 0..h {
            let tr = &tmp[i * w..(i + 1) * w];
            let dr = &mut dp[i * w..(i + 1) * w];
            for j in 0..w {
                let bj = &bw[j * w..(j + 1) * w];
                let mut acc = 0.0f32;
                for k in 0..w {
                    acc += tr[k] * bj[k];
                }
                dr[j] = acc;
            }
        }
    }
    dst
}

/// Обратное 2D ортонормированное DCT-II над плоскостями `[planes, H, W]`: `B_Hᵀ · C · B_W`.
fn idct2_planes(src: &[f32], planes: usize, h: usize, w: usize) -> Vec<f32> {
    let bh = dct2_basis(h);
    let bw = dct2_basis(w);
    let hw = h * w;
    let mut dst = vec![0f32; planes * hw];
    let mut tmp = vec![0f32; hw];
    for p in 0..planes {
        let sp = &src[p * hw..(p + 1) * hw];
        let dp = &mut dst[p * hw..(p + 1) * hw];
        tmp.fill(0.0);
        for k in 0..h {
            let bk = &bh[k * h..(k + 1) * h];
            let sr = &sp[k * w..(k + 1) * w];
            for i in 0..h {
                let bval = bk[i];
                let tr = &mut tmp[i * w..(i + 1) * w];
                for j in 0..w {
                    tr[j] += bval * sr[j];
                }
            }
        }
        for i in 0..h {
            let tr = &tmp[i * w..(i + 1) * w];
            let dr = &mut dp[i * w..(i + 1) * w];
            dr.fill(0.0);
            for k in 0..w {
                let tval = tr[k];
                let bk = &bw[k * w..(k + 1) * w];
                for j in 0..w {
                    dr[j] += tval * bk[j];
                }
            }
        }
    }
    dst
}

/// Понижение разрешения 5D-латента через 2D DCT low-pass (SPEED `_downsample_source`
/// с сохранением среднего уровня `sqrt(H_out·W_out / H_in·W_in)`).
fn lowpass_dct_5d(x: &Tensor, h_out: usize, w_out: usize) -> R<Tensor> {
    let d = x.dims().to_vec();
    let (b, c, t, h_in, w_in) = (d[0], d[1], d[2], d[3], d[4]);
    if h_in == h_out && w_in == w_out {
        return Ok(x.clone());
    }
    let dev = x.device();
    let dt = x.dtype();
    let src = x.to_device(Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let planes = b * c * t;
    let coeff = dct2_planes(&src, planes, h_in, w_in);
    let in_hw = h_in * w_in;
    let out_hw = h_out * w_out;
    let mut cropped = vec![0f32; planes * out_hw];
    for p in 0..planes {
        let sp = &coeff[p * in_hw..(p + 1) * in_hw];
        let dp = &mut cropped[p * out_hw..(p + 1) * out_hw];
        for y in 0..h_out {
            dp[y * w_out..(y + 1) * w_out].copy_from_slice(&sp[y * w_in..y * w_in + w_out]);
        }
    }
    let mut restored = idct2_planes(&cropped, planes, h_out, w_out);
    let gain = ((out_hw as f64) / (in_hw as f64)).sqrt() as f32;
    for v in &mut restored {
        *v *= gain;
    }
    Tensor::from_vec(restored, vec![b, c, t, h_out, w_out], dev)?.to_dtype(dt)
}

/// Спектральное расширение состояния 5D-латента (SPEED `spectral_expand`):
/// переносит НЧ-коэффициенты DCT-II грубого состояния в полный спектр, а новые
/// ВЧ-частоты заполняет гауссовым шумом амплитуды `sigma`, затем делает обратное
/// 2D DCT-II. Без размытия и потери дисперсии билинейного апскейла.
fn spectral_expand_5d(x: &Tensor, h_out: usize, w_out: usize, sigma: f32, seed: u64) -> R<Tensor> {
    let d = x.dims().to_vec();
    let (b, c, t, h_in, w_in) = (d[0], d[1], d[2], d[3], d[4]);
    if h_in == h_out && w_in == w_out {
        return Ok(x.clone());
    }
    let dev = x.device();
    let dt = x.dtype();
    let src = x.to_device(Device::Cpu)?.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
    let planes = b * c * t;
    let low_coeff = dct2_planes(&src, planes, h_in, w_in);
    let in_hw = h_in * w_in;
    let out_hw = h_out * w_out;

    let mut rng = synaptix_ops::rng::Philox4x32::new(seed);
    let mut full_coeff = vec![0f32; planes * out_hw];
    synaptix_ops::rng::fill_normal_f32(&mut rng, &mut full_coeff);
    for v in &mut full_coeff {
        *v *= sigma;
    }
    for p in 0..planes {
        let sp = &low_coeff[p * in_hw..(p + 1) * in_hw];
        let dp = &mut full_coeff[p * out_hw..(p + 1) * out_hw];
        for y in 0..h_in {
            dp[y * w_out..y * w_out + w_in].copy_from_slice(&sp[y * w_in..(y + 1) * w_in]);
        }
    }
    let expanded = idct2_planes(&full_coeff, planes, h_out, w_out);
    Tensor::from_vec(expanded, vec![b, c, t, h_out, w_out], dev)?.to_dtype(dt)
}

/// Выравнивание сигмы при переходе разрешения (SPEED `aligned_sigma`):
/// возвращает `(kappa, next_sigma)` так, что `expanded * kappa` имеет точный
/// баланс `(1 - next_sigma) · x0 + next_sigma · eps`.
fn speed_aligned_sigma(sigma: f64, ratio: f64) -> (f64, f64) {
    let kappa = ratio / (1.0 + (ratio - 1.0) * sigma);
    (kappa, sigma * kappa)
}

fn speed_stage2_sched(
    sched: &H3Scheduler,
    coarse_geo: Geometry,
    full_geo: Geometry,
    coarse_steps: usize,
) -> H3Scheduler {
    let mut out = sched.clone();
    if coarse_steps > 0 && coarse_steps < out.steps() {
        let q = out.video_sigma(coarse_steps);
        let ratio = ((full_geo.latent_h * full_geo.latent_w) as f64
            / (coarse_geo.latent_h * coarse_geo.latent_w) as f64)
            .sqrt();
        if q > 0.0 && q < 1.0 && ratio > 1.0 {
            let (_, next_sigma) = speed_aligned_sigma(q, ratio);
            out.sigmas[coarse_steps] = next_sigma;
        }
    }
    out
}

/// Подготовить `PreparedRun` под грубую геометрию SPEED без пересчёта `TokenRefiner`.
fn prepare_coarse(
    dit: &H3Dit,
    full_prep: &PreparedRun,
    req: &DenoiseRequest<'_>,
    coarse_geo: Geometry,
    sched: &H3Scheduler,
) -> Result<PreparedRun, H3Error> {
    let text_len = req.cond.context.dims()[1];
    let prefix_vt = req.prefix_video.as_ref().map(|t| t.dims().get(2).copied().unwrap_or(0)).unwrap_or(0);
    let prefix_at = req.prefix_audio.as_ref().map(|t| t.dims().get(3).copied().unwrap_or(0)).unwrap_or(0);
    let layout = PackedLayout::build(
        &LayoutRequest::new(text_len, coarse_geo.latent_t, coarse_geo.latent_h, coarse_geo.latent_w, coarse_geo.audio_t)
            .with_frame_count(coarse_geo.frame_count)
            .with_keyframes(req.keyframes.clone())
            .with_prefix(prefix_vt, prefix_at),
    )?;
    let plan = AdalnPlan::build(&layout, sched, req.visual_cond_aug, req.audio_cond_aug);
    let segments = mod_segments(&layout, &plan.roles, Some(&req.cond.text_tags));
    let rope = dit.rope_tables(&layout.positions)?;
    let vseg = layout.segment(SegmentKind::Video).ok_or_else(|| H3Error::Layout("нет video-сегмента".into()))?;
    let aseg = layout.segment(SegmentKind::Audio).ok_or_else(|| H3Error::Layout("нет audio-сегмента".into()))?;
    let vrow = plan.roles.index(plan.roles.role_for(SegmentKind::Video)) * crate::config::ADALN_MODALITIES;
    let arow = plan.roles.index(plan.roles.role_for(SegmentKind::Audio)) * crate::config::ADALN_MODALITIES;
    Ok(PreparedRun {
        layout,
        plan,
        segments,
        rope,
        video_seg: (vseg.start, vseg.stop, vrow / crate::config::ADALN_MODALITIES),
        audio_seg: (aseg.start, aseg.stop, arow / crate::config::ADALN_MODALITIES),
        refined_cond: full_prep.refined_cond.clone(),
        refined_negative: full_prep.refined_negative.clone(),
    })
}

/// Уменьшить патч-строки ключевых кадров (`cond_rows.video`) до грубой сетки SPEED.
fn resize_keyframe_cond_rows(
    rows: &Tensor,
    full_g: Geometry,
    coarse_g: Geometry,
    latents_dim: usize,
    patch: [usize; 3],
) -> R<Tensor> {
    let full_fr = (full_g.latent_h / patch[1]) * (full_g.latent_w / patch[2]);
    let n_kf = rows.dims()[0] / full_fr.max(1);
    let mut parts = Vec::with_capacity(n_kf);
    for i in 0..n_kf {
        let r = rows.narrow(0, i * full_fr, full_fr)?.contiguous()?;
        let lat = unpatchify_video(&r, 1, full_g.latent_h / patch[1], full_g.latent_w / patch[2], latents_dim, patch)?;
        let small = lowpass_dct_5d(&lat, coarse_g.latent_h, coarse_g.latent_w)?;
        parts.push(patchify_video(&small, patch)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, 0)
}

#[allow(clippy::too_many_arguments)]
fn is_oom(e: &impl std::fmt::Display) -> bool {
    let s = e.to_string();
    s.contains("OUT_OF_MEMORY") || s.contains("OOM") || s.contains("out of memory")
}

pub fn denoise_av(
    dit: &H3Dit,
    cache: &AdalnCache,
    prep: &PreparedRun,
    req: &DenoiseRequest<'_>,
    sched: &H3Scheduler,
    hooks: &DenoiseHooks<'_>,
) -> Result<DenoiseOutput, H3Error> {
    let cfg = &dit.cfg;
    let g = req.geometry;
    let device = dit.device();
    let compute = dit.compute_dtype();

    let (mut v_lat, mut a_lat) = match (&req.init_video, &req.init_audio) {
        (Some(v), Some(a)) => (v.clone(), a.clone()),
        _ => {
            let (v, a) = init_latents(
                g,
                cfg.latents_dim,
                cfg.audio_latents_dim,
                device,
                compute,
                req.seed,
            )?;
            (req.init_video.clone().unwrap_or(v), req.init_audio.clone().unwrap_or(a))
        }
    };

    check_cond_rows("видео", req.cond_rows.video.as_ref(), &prep.layout.img_update)?;
    check_cond_rows("аудио", req.cond_rows.audio.as_ref(), &prep.layout.audio_update)?;
    // Cond-строки приходят из VAE в F32 (зашумление считается в нём), целевые
    // идут в вычислительном типе DiT — приводим один раз, а не на каждом шаге.
    let cast = |t: &Option<Tensor>| -> R<Option<Tensor>> {
        t.as_ref().map(|c| c.to_device(device)?.to_dtype(compute)).transpose()
    };
    let cond_video = cast(&req.cond_rows.video)?;
    let cond_audio = cast(&req.cond_rows.audio)?;

    // Префикс-контекст (продолжение с хвоста): замороженные первые кадры видео и звука.
    let v_prefix: Option<(Tensor, Tensor, usize)> = match &req.prefix_video {
        Some(pv) => {
            let d = pv.dims();
            if d.len() != 5 || d[0] != 1 || d[1] != cfg.latents_dim || d[3] != g.latent_h || d[4] != g.latent_w {
                return Err(H3Error::Layout(format!(
                    "prefix_video {:?} не совпадает с целевой сеткой [1, {}, *, {}, {}]",
                    d, cfg.latents_dim, g.latent_h, g.latent_w
                )));
            }
            let pt = d[2];
            if pt == 0 || pt >= g.latent_t {
                return Err(H3Error::Layout(format!(
                    "prefix_video T={pt} вне допустимого диапазона 1..{}",
                    g.latent_t
                )));
            }
            let exact = pv.to_device(device)?.to_dtype(compute)?;
            let aug = req.visual_cond_aug.unwrap_or(VISUAL_COND_TIMESTEP);
            let noisy = if aug < 1.0 {
                let p32 = pv.to_device(device)?.to_dtype(DType::F32)?;
                apply_noise_aug(&p32, aug, req.seed.unwrap_or(0))?.to_dtype(compute)?
            } else {
                exact.clone()
            };
            Some((exact, noisy, pt))
        }
        None => None,
    };
    let a_prefix: Option<(Tensor, Tensor, usize)> = match &req.prefix_audio {
        Some(pa) => {
            let d = pa.dims();
            if d.len() != 4 || d[0] != 1 || d[1] != cfg.audio_latents_dim || d[2] != 2 {
                return Err(H3Error::Layout(format!(
                    "prefix_audio {:?} не совпадает с целевым форматом [1, {}, 2, *]",
                    d, cfg.audio_latents_dim
                )));
            }
            let pt = d[3];
            if pt == 0 || pt >= g.audio_t {
                return Err(H3Error::Layout(format!(
                    "prefix_audio T={pt} вне допустимого диапазона 1..{}",
                    g.audio_t
                )));
            }
            let exact = pa.to_device(device)?.to_dtype(compute)?;
            let aug = req.audio_cond_aug.unwrap_or(AUDIO_COND_TIMESTEP);
            let noisy = if aug < 1.0 {
                let p32 = pa.to_device(device)?.to_dtype(DType::F32)?;
                apply_noise_aug(&p32, aug, req.seed.unwrap_or(0).wrapping_add(1))?.to_dtype(compute)?
            } else {
                exact.clone()
            };
            Some((exact, noisy, pt))
        }
        None => None,
    };
    let splice_v = |cur: &Tensor, pref: &Tensor, pt: usize| -> R<Tensor> {
        let rest = cur.narrow(2, pt, g.latent_t - pt)?.contiguous()?;
        Tensor::cat(&[pref, &rest], 2)
    };
    let splice_a = |cur: &Tensor, pref: &Tensor, pt: usize| -> R<Tensor> {
        let rest = cur.narrow(3, pt, g.audio_t - pt)?.contiguous()?;
        Tensor::cat(&[pref, &rest], 3)
    };
    if let Some((_, ref p_in, pt)) = v_prefix {
        v_lat = splice_v(&v_lat, p_in, pt)?;
    }
    if let Some((_, ref p_in, pt)) = a_prefix {
        a_lat = splice_a(&a_lat, p_in, pt)?;
    }

    let steps = sched.steps();
    let patch = cfg.patch_size;

    // 2-стадийный спектральный coarse-to-fine семплинг (SPEED): первые шаги идут на
    // уменьшенной сетке (не ниже 34×60 латентов, чтобы держать кадрирование), затем
    // состояние расширяется в спектре 2D DCT-II с выравниванием сигмы `aligned_sigma`.
    if let Some((coarse_geo, coarse_steps)) = speed_plan(req, steps) {
        let q = sched.video_sigma(coarse_steps);
        let ratio = ((g.latent_h * g.latent_w) as f64
            / (coarse_geo.latent_h * coarse_geo.latent_w) as f64)
            .sqrt();
        let (kappa, next_sigma) = speed_aligned_sigma(q, ratio);
        let stage2_sched = speed_stage2_sched(sched, coarse_geo, g, coarse_steps);
        eprintln!(
            "[h3] SPEED (DCT coarse-to-fine): шаги 0..{coarse_steps} на {}×{} ({}×{} лат.) → шаги {coarse_steps}..{steps} на {}×{} ({}×{} лат., σ {q:.4} → {next_sigma:.4})",
            coarse_geo.width, coarse_geo.height, coarse_geo.latent_h, coarse_geo.latent_w,
            g.width, g.height, g.latent_h, g.latent_w
        );
        let coarse_prep = prepare_coarse(dit, prep, req, coarse_geo, sched)?;
        let (mut cv_lat, mut ca_lat) = init_latents(
            coarse_geo,
            cfg.latents_dim,
            cfg.audio_latents_dim,
            device,
            compute,
            req.seed,
        )?;
        let coarse_cond_v = match &cond_video {
            Some(cv) => Some(resize_keyframe_cond_rows(cv, g, coarse_geo, cfg.latents_dim, patch)?),
            None => None,
        };
        let coarse_v_prefix: Option<(Tensor, Tensor, usize)> = match &v_prefix {
            Some((exact, noisy, pt)) => Some((
                lowpass_dct_5d(exact, coarse_geo.latent_h, coarse_geo.latent_w)?,
                lowpass_dct_5d(noisy, coarse_geo.latent_h, coarse_geo.latent_w)?,
                *pt,
            )),
            None => None,
        };
        if let Some((_, ref p_in, pt)) = coarse_v_prefix {
            let rest = cv_lat.narrow(2, pt, coarse_geo.latent_t - pt)?.contiguous()?;
            cv_lat = Tensor::cat(&[p_in, &rest], 2)?;
        }
        if let Some((_, ref p_in, pt)) = a_prefix {
            ca_lat = splice_a(&ca_lat, p_in, pt)?;
        }
        let mut old_cv: Option<Tensor> = None;
        let mut old_ca: Option<Tensor> = None;
        dit.set_adapt_resident(false);
        run_denoise_range(
            dit,
            cache,
            &coarse_prep,
            req,
            sched,
            hooks,
            coarse_geo,
            0..coarse_steps,
            steps,
            coarse_cond_v.as_ref(),
            cond_audio.as_ref(),
            coarse_v_prefix.as_ref(),
            a_prefix.as_ref(),
            &mut cv_lat,
            &mut ca_lat,
            &mut old_cv,
            &mut old_ca,
            None,
        )?;
        drop(coarse_prep);
        drop(coarse_cond_v);
        drop(coarse_v_prefix);
        drop(old_cv);

        // Переход Stage 1 → Stage 2 (Spectral Progressive Diffusion):
        // переносим НЧ-спектр состояния `cv_lat` в 2D DCT-II полного разрешения,
        // ВЧ-полосу заполняем шумом `q · N(0, 1)`, затем умножаем на `kappa` —
        // баланс `(1 - next_sigma) · x0 + next_sigma · eps` сохраняется точно.
        let expanded_v = spectral_expand_5d(
            &cv_lat,
            g.latent_h,
            g.latent_w,
            q as f32,
            req.seed.unwrap_or(0).wrapping_add(10_000),
        )?;
        drop(cv_lat);
        v_lat = expanded_v.mul_scalar(kappa as f32)?;
        if let Some((_, ref p_in, pt)) = v_prefix {
            v_lat = splice_v(&v_lat, p_in, pt)?;
        }

        // Аудио `clock_reindex`: сдвигаем нативное состояние `ca_lat` с `old_sa` на `new_sa`.
        a_lat = if let Some(ref clean_a) = old_ca {
            let old_sa = sched.audio_sigma(coarse_steps);
            let new_sa = stage2_sched.audio_sigma(coarse_steps);
            if old_sa > 1e-6 && (new_sa - old_sa).abs() > 1e-6 {
                let noise_a = ca_lat
                    .sub(&clean_a.mul_scalar((1.0 - old_sa) as f32)?)?
                    .mul_scalar((1.0 / old_sa) as f32)?;
                clean_a
                    .mul_scalar((1.0 - new_sa) as f32)?
                    .add(&noise_a.mul_scalar(new_sa as f32)?)?
            } else {
                ca_lat
            }
        } else {
            ca_lat
        };
        if let Some((_, ref p_in, pt)) = a_prefix {
            a_lat = splice_a(&a_lat, p_in, pt)?;
        }
        let mut old_denoised_v: Option<Tensor> = None;
        let mut old_denoised_a: Option<Tensor> = old_ca;

        // Замер пика VRAM начинаем заново уже на полном разрешении.
        dit.set_adapt_resident(true);
        dit.reset_free_watch();
        run_denoise_range(
            dit,
            cache,
            prep,
            req,
            &stage2_sched,
            hooks,
            g,
            coarse_steps..steps,
            steps,
            cond_video.as_ref(),
            cond_audio.as_ref(),
            v_prefix.as_ref(),
            a_prefix.as_ref(),
            &mut v_lat,
            &mut a_lat,
            &mut old_denoised_v,
            &mut old_denoised_a,
            None,
        )?;
        if let Some((ref exact, _, pt)) = v_prefix {
            v_lat = splice_v(&v_lat, exact, pt)?;
        }
        if let Some((ref exact, _, pt)) = a_prefix {
            a_lat = splice_a(&a_lat, exact, pt)?;
        }
        hooks.report(steps, steps, stage2_sched.video_sigma(steps));
        return Ok(DenoiseOutput { video_latent: v_lat, audio_latent: a_lat });
    }

    let mut old_denoised_v: Option<Tensor> = None;
    let mut old_denoised_a: Option<Tensor> = None;

    // Раскладка блоков подстраивается по факту (H3Dit::adapt_resident).
    dit.reset_free_watch();
    run_denoise_range(
        dit,
        cache,
        prep,
        req,
        sched,
        hooks,
        g,
        0..steps,
        steps,
        cond_video.as_ref(),
        cond_audio.as_ref(),
        v_prefix.as_ref(),
        a_prefix.as_ref(),
        &mut v_lat,
        &mut a_lat,
        &mut old_denoised_v,
        &mut old_denoised_a,
        None,
    )?;
    if let Some((ref exact, _, pt)) = v_prefix {
        v_lat = splice_v(&v_lat, exact, pt)?;
    }
    if let Some((ref exact, _, pt)) = a_prefix {
        a_lat = splice_a(&a_lat, exact, pt)?;
    }
    hooks.report(steps, steps, sched.video_sigma(steps));

    Ok(DenoiseOutput { video_latent: v_lat, audio_latent: a_lat })
}

#[allow(clippy::too_many_arguments)]
fn run_denoise_range(
    dit: &H3Dit,
    cache: &AdalnCache,
    prep: &PreparedRun,
    req: &DenoiseRequest<'_>,
    sched: &H3Scheduler,
    hooks: &DenoiseHooks<'_>,
    g: Geometry,
    range: std::ops::Range<usize>,
    total_steps: usize,
    cond_video: Option<&Tensor>,
    cond_audio: Option<&Tensor>,
    v_prefix: Option<&(Tensor, Tensor, usize)>,
    a_prefix: Option<&(Tensor, Tensor, usize)>,
    v_lat: &mut Tensor,
    a_lat: &mut Tensor,
    old_denoised_v: &mut Option<Tensor>,
    old_denoised_a: &mut Option<Tensor>,
    mut last_denoised_v: Option<&mut Tensor>,
) -> Result<(), H3Error> {
    let cfg = &dit.cfg;
    let patch = cfg.patch_size;
    let splice_v = |cur: &Tensor, pref: &Tensor, pt: usize| -> R<Tensor> {
        let rest = cur.narrow(2, pt, g.latent_t - pt)?.contiguous()?;
        Tensor::cat(&[pref, &rest], 2)
    };
    let splice_a = |cur: &Tensor, pref: &Tensor, pt: usize| -> R<Tensor> {
        let rest = cur.narrow(3, pt, g.audio_t - pt)?.contiguous()?;
        Tensor::cat(&[pref, &rest], 3)
    };
    if let Some(sp) = sparse_cfg_for_step(range.end.saturating_sub(1), total_steps, prep) {
        let dense_n = range.clone().take_while(|&s| sparse_cfg_for_step(s, total_steps, prep).is_none()).count();
        eprintln!(
            "[h3] разреженное FP8-внимание (Sol): плотных шагов {dense_n}, далее keep={:.2}, защита 0..{} + ±{} ток.",
            sp.keep_ratio, sp.protected_prefix, sp.frame_tokens
        );
    }
    for step in range {
        if hooks.cancelled() {
            return Err(H3Error::Cancelled);
        }
        hooks.report(step, total_steps, sched.video_sigma(step));

        if step == dump_step() {
            dump_tensor("step_v_lat", v_lat);
            dump_tensor("step_a_lat", a_lat);
            dump_text("step_sigma", &format!("{}", sched.video_sigma(step)));
        }
        let v_rows_t = patchify_video(v_lat, patch)?;
        let a_rows_t = pack_audio(a_lat)?;
        let v_rows = merge_stream_rows(&v_rows_t, cond_video, &prep.layout.img_update)?;
        let a_rows = merge_stream_rows(&a_rows_t, cond_audio, &prep.layout.audio_update)?;

        let v_emb = dit.embed_video(&v_rows)?;
        let a_emb = dit.embed_audio(&a_rows)?;
        drop(v_rows_t);
        drop(a_rows_t);
        drop(v_rows);
        drop(a_rows);

        let hidden = assemble_hidden(prep, &prep.refined_cond, &v_emb, &a_emb)?;
        let need_neg = prep.refined_negative.is_some() && req.guider.needs_uncond(step);
        let embs = if need_neg {
            Some((v_emb, a_emb))
        } else {
            drop(v_emb);
            drop(a_emb);
            None
        };
        if step == 0 && runtime::h3_prof() {
            for (bytes, count) in synaptix_core::memory::cuda_pool::live_alloc_top(8) {
                eprintln!("[pre-fwd] {bytes} B × {count}");
            }
        }
        // Нехватка видеопамяти — поднятые по ходу блоки обратно на хост и шаг заново.
        let _sparse = synaptix_core::precision::AttentionSparseGuard::new(sparse_cfg_for_step(step, total_steps, prep));
        let fwd = |h: &Tensor| loop {
            match dit.forward(h, cache, step, &prep.segments, &prep.rope, prep.video_seg, prep.audio_seg) {
                Err(e) if is_oom(&e) && dit.shed_promoted() => continue,
                r => break r,
            }
        };
        let (v_out, a_out) = fwd(&hidden)?;
        drop(hidden);

        let (v_out, a_out) = match (&prep.refined_negative, &embs) {
            (Some(neg), Some((v_emb, a_emb))) => {
                let n_hidden = assemble_hidden(prep, neg, v_emb, a_emb)?;
                let (nv, na) = fwd(&n_hidden)?;
                (
                    apply_cfg(&v_out, &nv, req.guider.cfg_scale)?,
                    apply_cfg(&a_out, &na, req.guider.cfg_scale)?,
                )
            }
            _ => (v_out, a_out),
        };
        drop(_sparse);

        let v_vel = unpatchify_video(
            &v_out.mul_scalar(-1.0)?,
            g.latent_t / patch[0],
            g.latent_h / patch[1],
            g.latent_w / patch[2],
            cfg.latents_dim,
            patch,
        )?;
        let a_vel = unpack_audio(&a_out.mul_scalar(-1.0)?, 2)?;

        let sv = sched.video_sigma(step);
        let sa = sched.audio_sigma(step);
        let denoised_v = v_lat.sub(&v_vel.to_dtype(v_lat.dtype())?.mul_scalar(sv as f32)?)?;
        let denoised_a = a_lat.sub(&a_vel.to_dtype(a_lat.dtype())?.mul_scalar(sa as f32)?)?;

        match req.sampler {
            SamplerKind::Euler => {
                *v_lat = v_lat.add(&v_vel.to_dtype(v_lat.dtype())?.mul_scalar(sched.video_dt(step) as f32)?)?;
                *a_lat = a_lat.add(&a_vel.to_dtype(a_lat.dtype())?.mul_scalar(sched.audio_dt(step) as f32)?)?;
            }
            SamplerKind::ResMultistep => {
                *v_lat = res_multistep_update(
                    v_lat,
                    &v_vel,
                    &denoised_v,
                    old_denoised_v.as_ref(),
                    if step > 0 { Some(sched.video_sigma(step - 1)) } else { None },
                    sv,
                    sched.video_sigma(step + 1),
                )?;
                *a_lat = res_multistep_update(
                    a_lat,
                    &a_vel,
                    &denoised_a,
                    old_denoised_a.as_ref(),
                    if step > 0 { Some(sched.audio_sigma(step - 1)) } else { None },
                    sa,
                    sched.audio_sigma(step + 1),
                )?;
            }
        }
        if let Some((_, ref p_in, pt)) = v_prefix {
            *v_lat = splice_v(v_lat, p_in, *pt)?;
        }
        if let Some((_, ref p_in, pt)) = a_prefix {
            *a_lat = splice_a(a_lat, p_in, *pt)?;
        }
        let denoised_v = match v_prefix {
            Some((exact, _, pt)) => splice_v(&denoised_v, exact, *pt)?,
            None => denoised_v,
        };
        let denoised_a = match a_prefix {
            Some((exact, _, pt)) => splice_a(&denoised_a, exact, *pt)?,
            None => denoised_a,
        };
        if let Some(dst) = last_denoised_v.as_deref_mut() {
            *dst = denoised_v.clone();
        }
        *old_denoised_v = Some(denoised_v.clone());
        *old_denoised_a = Some(denoised_a);

        if runtime::h3_prof() {
            eprintln!(
                "[h3-prof] шаг {step}: σv {sv:.4} · {} · {} · {}",
                tensor_stats("v_vel", &v_vel),
                tensor_stats("v_lat", v_lat),
                tensor_stats("x0_video", &denoised_v)
            );
            eprintln!(
                "[h3-prof] шаг {step}: σa {sa:.4} · {} · {}",
                tensor_stats("a_vel", &a_vel),
                tensor_stats("a_lat", a_lat)
            );
        }
    }
    Ok(())
}

fn res_multistep_update(
    x: &Tensor,
    vel: &Tensor,
    denoised: &Tensor,
    old_denoised: Option<&Tensor>,
    sigma_prev: Option<f64>,
    sigma: f64,
    sigma_next: f64,
) -> R<Tensor> {
    match (old_denoised, sigma_prev) {
        (Some(old), Some(sp)) if sigma_next > 0.0 => {
            let t = -sigma.ln();
            let t_next = -sigma_next.ln();
            let t_prev = -sp.ln();
            let h = t_next - t;
            let c2 = (t_prev - t) / h;
            let phi1 = (-h).exp_m1() / (-h);
            let phi2 = (phi1 - 1.0) / (-h);
            let b1 = phi1 - phi2 / c2;
            let b2 = phi2 / c2;
            x.mul_scalar((-h).exp() as f32)?
                .add(&denoised.mul_scalar((h * b1) as f32)?)?
                .add(&old.mul_scalar((h * b2) as f32)?)
        }
        _ => x.add(&vel.to_dtype(x.dtype())?.mul_scalar((sigma_next - sigma) as f32)?),
    }
}

pub fn cond_rows_from_keyframe_latents(
    latents: &[Tensor],
    patch: [usize; 3],
    noise_aug: Option<f32>,
    seed: u64,
) -> Result<Option<Tensor>, H3Error> {
    if latents.is_empty() {
        return Ok(None);
    }
    let aug = noise_aug.unwrap_or(VISUAL_COND_TIMESTEP);
    let mut rows = Vec::with_capacity(latents.len());
    for z in latents {
        let r = patchify_video(&z.to_dtype(DType::F32)?, patch)?;
        rows.push(apply_noise_aug(&r, aug, seed)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Some(Tensor::cat(&refs, 0)?))
}

pub fn cond_rows_from_audio_latents(
    latents: &[Tensor],
    noise_aug: Option<f32>,
    seed: u64,
) -> Result<Option<Tensor>, H3Error> {
    if latents.is_empty() {
        return Ok(None);
    }
    let aug = noise_aug.unwrap_or(AUDIO_COND_TIMESTEP);
    let mut rows = Vec::with_capacity(latents.len());
    for z in latents {
        let r = pack_audio(&z.to_dtype(DType::F32)?)?;
        rows.push(apply_noise_aug(&r, aug, seed + 1)?);
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Ok(Some(Tensor::cat(&refs, 0)?))
}

pub(crate) fn apply_noise_aug(rows: &Tensor, aug: f32, seed: u64) -> Result<Tensor, H3Error> {
    if aug >= 1.0 {
        return Ok(rows.clone());
    }
    let mut rng = synaptix_ops::rng::Philox4x32::new(seed);
    let noise = randn(rows.dims(), rows.device(), rows.dtype(), &mut rng)?;
    Ok(rows.mul_scalar(aug)?.add(&noise.mul_scalar(1.0 - aug)?)?)
}

#[cfg(test)]
mod speed_tests {
    use super::*;

    #[test]
    fn speed_coarse_dims_respects_floor() {
        assert_eq!(speed_coarse_dims(1344, 768), Some((960, 544)));
        assert_eq!(speed_coarse_dims(1920, 1088), Some((960, 544)));
        assert_eq!(speed_coarse_dims(640, 384), None);
    }

    #[test]
    fn dct2_idct2_roundtrip_and_spectral_expand() {
        let data: Vec<f32> = (0..2 * 4 * 6).map(|i| ((i as f32) * 0.37).sin()).collect();
        let c = dct2_planes(&data, 2, 4, 6);
        let rec = idct2_planes(&c, 2, 4, 6);
        let max_err = data.iter().zip(&rec).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "dct2/idct2 roundtrip max_err={max_err}");

        // Константный x0 при q=0.5, ratio=2.0: после spectral_expand(sigma=0) * kappa
        // сигнал должен иметь вес (1 - next_sigma) точно.
        let q = 0.5f64;
        let ratio = 2.0f64;
        let (kappa, next_sigma) = speed_aligned_sigma(q, ratio);
        let x = Tensor::from_vec(vec![(1.0 - q) as f32; 2 * 4 * 6], vec![1, 1, 2, 4, 6], Device::Cpu).unwrap();
        let y = spectral_expand_5d(&x, 8, 12, 0.0, 42).unwrap();
        assert_eq!(y.dims(), &[1, 1, 2, 8, 12]);
        let v: Vec<f32> = y.flatten_all().unwrap().to_vec1::<f32>().unwrap().into_iter().map(|a| a * kappa as f32).collect();
        let want = (1.0 - next_sigma) as f32;
        assert!(v.iter().all(|&a| (a - want).abs() < 1e-4), "want {want}, got {}", v[0]);

        let down = lowpass_dct_5d(&y, 4, 6).unwrap();
        let dv: Vec<f32> = down.flatten_all().unwrap().to_vec1::<f32>().unwrap().into_iter().map(|a| a * kappa as f32).collect();
        assert!(dv.iter().all(|&a| (a - want).abs() < 1e-4));
    }
}

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
    let layout = PackedLayout::build(
        &LayoutRequest::new(text_len, g.latent_t, g.latent_h, g.latent_w, g.audio_t)
            .with_frame_count(g.frame_count)
            .with_keyframes(req.keyframes.clone())
            .with_refs(req.refs.clone()),
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
    let plan = AdalnPlan::build(&layout, sched, req.visual_cond_aug, req.audio_cond_aug);
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
    let v_rows =
        merge_stream_rows(&v_rows_t, req.cond_rows.video.as_ref(), &prep.layout.img_update)?;
    let a_rows =
        merge_stream_rows(&a_rows_t, req.cond_rows.audio.as_ref(), &prep.layout.audio_update)?;
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

#[allow(clippy::too_many_arguments)]
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

    let steps = sched.steps();
    let patch = cfg.patch_size;
    let mut old_denoised_v: Option<Tensor> = None;
    let mut old_denoised_a: Option<Tensor> = None;

    for step in 0..steps {
        if hooks.cancelled() {
            return Err(H3Error::Cancelled);
        }
        hooks.report(step, steps, sched.video_sigma(step));

        if step == dump_step() {
            dump_tensor("step_v_lat", &v_lat);
            dump_tensor("step_a_lat", &a_lat);
            dump_text("step_sigma", &format!("{}", sched.video_sigma(step)));
        }
        let v_rows_t = patchify_video(&v_lat, patch)?;
        let a_rows_t = pack_audio(&a_lat)?;
        let v_rows = merge_stream_rows(&v_rows_t, req.cond_rows.video.as_ref(), &prep.layout.img_update)?;
        let a_rows = merge_stream_rows(&a_rows_t, req.cond_rows.audio.as_ref(), &prep.layout.audio_update)?;

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
        let (v_out, a_out) = dit.forward(
            &hidden,
            cache,
            step,
            &prep.segments,
            &prep.rope,
            prep.video_seg,
            prep.audio_seg,
        )?;
        drop(hidden);

        let (v_out, a_out) = match (&prep.refined_negative, &embs) {
            (Some(neg), Some((v_emb, a_emb))) => {
                let n_hidden = assemble_hidden(prep, neg, v_emb, a_emb)?;
                let (nv, na) = dit.forward(
                    &n_hidden,
                    cache,
                    step,
                    &prep.segments,
                    &prep.rope,
                    prep.video_seg,
                    prep.audio_seg,
                )?;
                (
                    apply_cfg(&v_out, &nv, req.guider.cfg_scale)?,
                    apply_cfg(&a_out, &na, req.guider.cfg_scale)?,
                )
            }
            _ => (v_out, a_out),
        };

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
                v_lat = v_lat.add(&v_vel.to_dtype(v_lat.dtype())?.mul_scalar(sched.video_dt(step) as f32)?)?;
                a_lat = a_lat.add(&a_vel.to_dtype(a_lat.dtype())?.mul_scalar(sched.audio_dt(step) as f32)?)?;
            }
            SamplerKind::ResMultistep => {
                v_lat = res_multistep_update(
                    &v_lat,
                    &v_vel,
                    &denoised_v,
                    old_denoised_v.as_ref(),
                    if step > 0 { Some(sched.video_sigma(step - 1)) } else { None },
                    sv,
                    sched.video_sigma(step + 1),
                )?;
                a_lat = res_multistep_update(
                    &a_lat,
                    &a_vel,
                    &denoised_a,
                    old_denoised_a.as_ref(),
                    if step > 0 { Some(sched.audio_sigma(step - 1)) } else { None },
                    sa,
                    sched.audio_sigma(step + 1),
                )?;
            }
        }
        old_denoised_v = Some(denoised_v.clone());
        old_denoised_a = Some(denoised_a);

        if runtime::h3_prof() {
            eprintln!(
                "[h3-prof] шаг {step}: σv {sv:.4} · {} · {} · {}",
                tensor_stats("v_vel", &v_vel),
                tensor_stats("v_lat", &v_lat),
                tensor_stats("x0_video", &denoised_v)
            );
            eprintln!(
                "[h3-prof] шаг {step}: σa {sa:.4} · {} · {}",
                tensor_stats("a_vel", &a_vel),
                tensor_stats("a_lat", &a_lat)
            );
        }
    }
    hooks.report(steps, steps, sched.video_sigma(steps));

    Ok(DenoiseOutput { video_latent: v_lat, audio_latent: a_lat })
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

fn apply_noise_aug(rows: &Tensor, aug: f32, seed: u64) -> Result<Tensor, H3Error> {
    if aug >= 1.0 {
        return Ok(rows.clone());
    }
    let mut rng = synaptix_ops::rng::Philox4x32::new(seed);
    let noise = randn(rows.dims(), rows.device(), rows.dtype(), &mut rng)?;
    Ok(rows.mul_scalar(aug)?.add(&noise.mul_scalar(1.0 - aug)?)?)
}

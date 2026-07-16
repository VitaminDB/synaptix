
use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::tokenizer::Tokenizer;

use crate::ar::CodesGenOptions;
use crate::cond_encoder::ConditionEncoder;
use crate::config::DitConfig;
use crate::detokenizer::Detokenizer;
use crate::dit::Dit;
use crate::fsq::Fsq;
use crate::lm::AceStepLm;
use crate::loader::{read_bundle_file, CompLoader};
use crate::scheduler::timestep_schedule;
use crate::text_encoder::TextEncoder;
use crate::tokenizer::{AceTokenizer, Metadata};
use crate::vae::AceStepVae;
use crate::AceError;

#[derive(Debug, Clone)]
pub struct SamplerOptions {
    pub steps: usize,
    pub shift: f32,
    pub guidance_scale: f32,
    pub dcw: crate::dcw::DcwCorrector,
}

impl Default for SamplerOptions {
    fn default() -> Self {
        Self {
            steps: 8,
            shift: 3.0,
            guidance_scale: 7.0,
            dcw: crate::dcw::DcwCorrector::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum EditMode {
    #[default]
    Text2Music,
    Retake,
    Repaint,
    Edit,
    Extract,
}

#[derive(Debug, Clone, Default)]
pub struct EditOptions {
    pub mode: EditMode,
    pub retake_variance: f32,
    pub retake_seed: u64,
    pub src_latent: Option<Tensor>,
    pub repaint_start_sec: f32,
    pub repaint_end_sec: f32,
    pub repaint_strength: f32,
    pub edit_n_min: f32,
    pub edit_n_max: f32,
    pub edit_n_avg: usize,
    pub edit_source_caption: String,
    pub edit_source_lyric: String,
}

pub struct ApgBuffer {
    running: Option<Tensor>,
    momentum: f32,
}

impl ApgBuffer {
    pub fn new() -> Self {
        Self { running: None, momentum: -0.75 }
    }
    fn update(&mut self, diff: &Tensor) -> Result<Tensor, AceError> {
        let r = match &self.running {
            Some(p) => diff.broadcast_add(&p.affine(self.momentum, 0.0)?)?,
            None => diff.clone(),
        };
        self.running = Some(r.clone());
        Ok(r)
    }
}

impl Default for ApgBuffer {
    fn default() -> Self {
        Self::new()
    }
}

pub fn apg_forward(
    pred_cond: &Tensor,
    pred_uncond: &Tensor,
    guidance_scale: f32,
    buf: &mut ApgBuffer,
) -> Result<Tensor, AceError> {
    const NORM_THR: f32 = 2.5;
    let out_dtype = pred_cond.dtype();
    let out_dev = pred_cond.device();
    let cond = pred_cond.to_device(Device::Cpu)?.to_dtype(DType::F64)?;
    let uncond = pred_uncond.to_device(Device::Cpu)?.to_dtype(DType::F64)?;
    let diff = cond.broadcast_add(&uncond.affine(-1.0, 0.0)?)?;
    let diff = buf.update(&diff)?;
    let dnorm = diff.sqr()?.sum_keepdim(1)?.sqrt()?;
    let scale = dnorm.affine(1.0, 1e-12)?.recip()?.affine(NORM_THR, 0.0)?.clamp(0.0, 1.0)?;
    let diff = diff.broadcast_mul(&scale)?;
    let cn = cond.sqr()?.sum_keepdim(1)?.sqrt()?.affine(1.0, 1e-12)?.recip()?;
    let v1 = cond.broadcast_mul(&cn)?;
    let par_coef = diff.broadcast_mul(&v1)?.sum_keepdim(1)?;
    let parallel = v1.broadcast_mul(&par_coef)?;
    let orthogonal = diff.broadcast_add(&parallel.affine(-1.0, 0.0)?)?;
    let guided = cond.broadcast_add(&orthogonal.affine(guidance_scale - 1.0, 0.0)?)?;
    Ok(guided.to_device(out_dev)?.to_dtype(out_dtype)?)
}

pub fn denoise(
    dit: &Dit,
    x_init: &Tensor,
    context_latents: &Tensor,
    enc_cond: &Tensor,
    enc_null: Option<&Tensor>,
    opts: &SamplerOptions,
) -> Result<Tensor, AceError> {
    let t = timestep_schedule(opts.steps, opts.shift);
    let cfg = opts.guidance_scale > 1.0 && enc_null.is_some();
    let mut apg = ApgBuffer::new();
    let mut x = x_init.clone();
    let (ctx2, enc2) = if cfg {
        (
            Some(Tensor::cat(&[context_latents, context_latents], 0)?),
            Some(Tensor::cat(&[enc_cond, enc_null.unwrap()], 0)?),
        )
    } else {
        (None, None)
    };

    let graph_ok = x_init.dims()[1] <= 8192;
    if cfg && !opts.dcw.is_active() && !dit.is_quantized() && graph_ok {
        if let synaptix_core::device::Device::Cuda(ord) = x_init.device() {
            return denoise_graph(
                dit,
                x_init,
                ctx2.as_ref().unwrap(),
                enc2.as_ref().unwrap(),
                &t,
                opts.guidance_scale,
                &mut apg,
                ord,
            );
        }
    }

    for i in 0..opts.steps {
        let tc = t[i];
        let v = if cfg {
            let x2 = Tensor::cat(&[&x, &x], 0)?;
            let v2 = dit.forward(&x2, tc, tc, ctx2.as_ref().unwrap(), enc2.as_ref().unwrap())?;
            let v_cond = v2.narrow(0, 0, 1)?.contiguous()?;
            let v_null = v2.narrow(0, 1, 1)?.contiguous()?;
            apg_forward(&v_cond, &v_null, opts.guidance_scale, &mut apg)?
        } else {
            dit.forward(&x, tc, tc, context_latents, enc_cond)?
        };
        let dt = t[i] - t[i + 1];
        let x_next = x.broadcast_add(&v.affine(-dt, 0.0)?)?;
        x = if opts.dcw.is_active() {
            let denoised = x.broadcast_add(&v.affine(-tc, 0.0)?)?;
            opts.dcw.apply(&x_next, &denoised, tc)?
        } else {
            x_next
        };
    }
    Ok(x)
}

#[allow(clippy::too_many_arguments)]
fn denoise_repaint(
    dit: &Dit,
    x_init: &Tensor,
    context_latents: &Tensor,
    enc_cond: &Tensor,
    enc_null: Option<&Tensor>,
    opts: &SamplerOptions,
    src: &Tensor,
    noise: &Tensor,
    mask: &Tensor,
    soft_mask: &Tensor,
    injection_cutoff: usize,
) -> Result<Tensor, AceError> {
    let t = timestep_schedule(opts.steps, opts.shift);
    let cfg = opts.guidance_scale > 1.0 && enc_null.is_some();
    let mut apg = ApgBuffer::new();
    let mut x = x_init.clone();
    let inv_mask = mask.affine(-1.0, 1.0)?;
    let (ctx2, enc2) = if cfg {
        (
            Some(Tensor::cat(&[context_latents, context_latents], 0)?),
            Some(Tensor::cat(&[enc_cond, enc_null.unwrap()], 0)?),
        )
    } else {
        (None, None)
    };
    for i in 0..opts.steps {
        let tc = t[i];
        let v = if cfg {
            let x2 = Tensor::cat(&[&x, &x], 0)?;
            let v2 = dit.forward(&x2, tc, tc, ctx2.as_ref().unwrap(), enc2.as_ref().unwrap())?;
            let v_cond = v2.narrow(0, 0, 1)?.contiguous()?;
            let v_null = v2.narrow(0, 1, 1)?.contiguous()?;
            apg_forward(&v_cond, &v_null, opts.guidance_scale, &mut apg)?
        } else {
            dit.forward(&x, tc, tc, context_latents, enc_cond)?
        };
        let dt = t[i] - t[i + 1];
        let x_next = x.broadcast_add(&v.affine(-dt, 0.0)?)?;
        x = if i < injection_cutoff {
            let tn = t[i + 1];
            let zt_src = noise.affine(tn, 0.0)?.broadcast_add(&src.affine(1.0 - tn, 0.0)?)?;
            x_next
                .broadcast_mul(mask)?
                .broadcast_add(&zt_src.broadcast_mul(&inv_mask)?)?
        } else {
            x_next
        };
    }
    let inv_soft = soft_mask.affine(-1.0, 1.0)?;
    let out = x.broadcast_mul(soft_mask)?.broadcast_add(&src.broadcast_mul(&inv_soft)?)?;
    Ok(out)
}

fn align_latent_len(src: &Tensor, t_frames: usize, device: Device) -> Result<Tensor, AceError> {
    let d = src.dims().to_vec();
    let ts = d[1];
    let src = src.to_device(device)?.to_dtype(DType::F32)?;
    if ts == t_frames {
        Ok(src)
    } else if ts > t_frames {
        Ok(src.narrow(1, 0, t_frames)?)
    } else {
        let pad = Tensor::zeros(vec![1usize, t_frames - ts, d[2]], DType::F32, device)?;
        Ok(Tensor::cat(&[&src, &pad], 1)?)
    }
}

fn build_repaint_masks(
    t_frames: usize,
    start_sec: f32,
    end_sec: f32,
    crossfade_frames: usize,
    device: Device,
) -> Result<(Tensor, Tensor), AceError> {
    const LATENT_HZ: f32 = 25.0;
    let start = ((start_sec.max(0.0) * LATENT_HZ).floor() as usize).min(t_frames.saturating_sub(1));
    let end = if end_sec < 0.0 {
        t_frames
    } else {
        ((end_sec * LATENT_HZ).floor() as usize).clamp(start + 1, t_frames)
    };
    let mut mask = vec![0.0f32; t_frames];
    for m in mask.iter_mut().take(end).skip(start) {
        *m = 1.0;
    }
    let mut soft = mask.clone();
    let cf = crossfade_frames.min(t_frames);
    for k in 0..cf {
        let val = (cf - k) as f32 / (cf as f32 + 1.0);
        if start > k {
            soft[start - 1 - k] = val;
        }
        let r = end + k;
        if r < t_frames {
            soft[r] = val;
        }
    }
    let mask_t = Tensor::from_vec(mask, vec![1usize, t_frames, 1], device)?;
    let soft_t = Tensor::from_vec(soft, vec![1usize, t_frames, 1], device)?;
    Ok((mask_t, soft_t))
}

#[allow(clippy::too_many_arguments)]
fn denoise_sdedit(
    dit: &Dit,
    src: &Tensor,
    noise: &Tensor,
    context_latents: &Tensor,
    enc_cond: &Tensor,
    enc_null: Option<&Tensor>,
    opts: &SamplerOptions,
    t_start: f32,
) -> Result<Tensor, AceError> {
    let t = timestep_schedule(opts.steps, opts.shift);
    let cfg = opts.guidance_scale > 1.0 && enc_null.is_some();
    let mut apg = ApgBuffer::new();
    let t_start = t_start.clamp(0.0, 1.0);
    let start_idx = (0..opts.steps).find(|&i| t[i] <= t_start).unwrap_or(opts.steps);
    let ts = if start_idx < opts.steps { t[start_idx] } else { 0.0 };
    let mut x = noise.affine(ts, 0.0)?.broadcast_add(&src.affine(1.0 - ts, 0.0)?)?;
    let (ctx2, enc2) = if cfg {
        (
            Some(Tensor::cat(&[context_latents, context_latents], 0)?),
            Some(Tensor::cat(&[enc_cond, enc_null.unwrap()], 0)?),
        )
    } else {
        (None, None)
    };
    for i in start_idx..opts.steps {
        let tc = t[i];
        let v = if cfg {
            let x2 = Tensor::cat(&[&x, &x], 0)?;
            let v2 = dit.forward(&x2, tc, tc, ctx2.as_ref().unwrap(), enc2.as_ref().unwrap())?;
            let v_cond = v2.narrow(0, 0, 1)?.contiguous()?;
            let v_null = v2.narrow(0, 1, 1)?.contiguous()?;
            apg_forward(&v_cond, &v_null, opts.guidance_scale, &mut apg)?
        } else {
            dit.forward(&x, tc, tc, context_latents, enc_cond)?
        };
        let dt = t[i] - t[i + 1];
        x = x.broadcast_add(&v.affine(-dt, 0.0)?)?;
    }
    Ok(x)
}

#[allow(clippy::too_many_arguments)]
fn denoise_graph(
    dit: &Dit,
    x0: &Tensor,
    ctx2: &Tensor,
    enc2: &Tensor,
    t: &[f32],
    guidance_scale: f32,
    apg: &mut ApgBuffer,
    ord: usize,
) -> Result<Tensor, AceError> {
    use synaptix_core::grad::no_grad;
    use synaptix_infer::error::InferError;
    use synaptix_infer::graph_capture::GraphCapturer;

    let device = x0.device();
    let steps = t.len() - 1;
    let stream = synaptix_core::device::cuda::default_stream(ord)
        .map_err(|e| AceError::Other(format!("default_stream: {e}")))?;

    let mut x = x0.clone();
    let (h0, orig_len) = dit.proj_in_h(&Tensor::cat(&[&x, &x], 0)?, ctx2)?;
    let s = h0.dims()[1];
    let (enc, cos, sin, sliding_mask) = dit.layer_inputs(enc2, s, device)?;

    let mut h_buf = h0;
    let (temb0, tproj0) = dit.compute_temb(t[0], t[0], device)?;
    let mut temb_buf = temb0;
    let mut tproj_buf = tproj0;
    let mut hn_buf = Tensor::zeros(h_buf.dims().to_vec(), h_buf.dtype(), device)?;

    let mut cap = GraphCapturer::new(3);
    let graph = {
        let d = dit;
        let (hb, tb, pb) = (&h_buf, &temb_buf, &tproj_buf);
        let (er, cr, sr, mr) = (&enc, &cos, &sin, &sliding_mask);
        let ob = &mut hn_buf;
        no_grad(|| {
            cap.capture_with(&stream, |_| {
                let hn = d
                    .forward_layers(hb, tb, pb, er, cr, sr, mr)
                    .map_err(|e| InferError::Other(e.to_string()))?;
                ob.copy_from(&hn).map_err(|e| InferError::Other(e.to_string()))
            })
        })
        .map_err(|e| AceError::Other(format!("dit graph capture: {e}")))?
    };
    graph.upload().map_err(|e| AceError::Other(format!("dit graph upload: {e}")))?;

    for i in 0..steps {
        let (h_i, _) = dit.proj_in_h(&Tensor::cat(&[&x, &x], 0)?, ctx2)?;
        h_buf.copy_from(&h_i)?;
        let (temb_i, tproj_i) = dit.compute_temb(t[i], t[i], device)?;
        temb_buf.copy_from(&temb_i)?;
        tproj_buf.copy_from(&tproj_i)?;
        graph.launch().map_err(|e| AceError::Other(format!("dit graph launch: {e:?}")))?;
        stream.synchronize().map_err(|e| AceError::Other(format!("dit graph sync: {e:?}")))?;
        let v2 = dit.proj_out_v(&hn_buf, orig_len)?;
        let v_cond = v2.narrow(0, 0, 1)?.contiguous()?;
        let v_null = v2.narrow(0, 1, 1)?.contiguous()?;
        let v = apg_forward(&v_cond, &v_null, guidance_scale, apg)?;
        let dt = t[i] - t[i + 1];
        x = x.broadcast_add(&v.affine(-dt, 0.0)?)?;
    }
    Ok(x)
}

pub fn peak_normalize(samples: &mut [f32]) {
    let peak = samples.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if peak < 1e-6 {
        return;
    }
    let gain = 0.891_250_94_f32 / peak;
    for x in samples.iter_mut() {
        *x *= gain;
    }
}

/// RMS normalize to ~-20 dBFS (0.1), then guard against clipping by capping the
/// peak at 0.99 — the loudness-matching counterpart to peak normalization.
pub fn rms_normalize(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let ms: f64 = samples.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / samples.len() as f64;
    let rms = ms.sqrt() as f32;
    if rms < 1e-6 {
        return;
    }
    let mut gain = 0.1_f32 / rms;
    let peak = samples.iter().fold(0f32, |m, &x| m.max(x.abs()));
    if peak * gain > 0.99 {
        gain = 0.99 / peak;
    }
    for x in samples.iter_mut() {
        *x *= gain;
    }
}

/// Apply the selected output normalization mode.
pub fn apply_norm(samples: &mut [f32], mode: NormMode) {
    match mode {
        NormMode::Off => {}
        NormMode::Peak => peak_normalize(samples),
        NormMode::Rms => rms_normalize(samples),
    }
}

/// Read the silence latent raw f32 + shape from the DiT `.syn` bundle, accepting
/// either `silence_latent.safetensors` (base) or `silence_latent.pt` (turbo —
/// a torch.save ZIP; see [`read_pt_silence`]).
fn load_silence_raw(dit_path: &Path) -> Result<(Vec<f32>, Vec<usize>), AceError> {
    if let Ok(bytes) = read_bundle_file(dit_path, "silence_latent.safetensors") {
        let st = safetensors::SafeTensors::deserialize(&bytes)
            .map_err(|e| AceError::Load(format!("silence_latent deserialize: {e}")))?;
        let name = if st.names().iter().any(|n| *n == "silence_latent") {
            "silence_latent".to_string()
        } else {
            st.names().first().map(|s| s.to_string())
                .ok_or_else(|| AceError::Load("silence_latent: пустой safetensors".into()))?
        };
        let view = st.tensor(&name).map_err(|e| AceError::Load(format!("silence_latent tensor: {e}")))?;
        let shape = view.shape().to_vec();
        let all: Vec<f32> = view.data().chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect();
        return Ok((all, shape));
    }
    let bytes = read_bundle_file(dit_path, "silence_latent.pt")?;
    read_pt_silence(&bytes)
}

/// Minimal reader for `silence_latent.pt` (torch.save = a ZIP archive, STORED).
/// Picks the largest STORED entry (the tensor storage `*/data/0`) from the
/// central directory and reads it as f32. The silence latent is the fixed
/// channel-major `[1, 64, T]` layout (the only tensor this loads).
fn read_pt_silence(bytes: &[u8]) -> Result<(Vec<f32>, Vec<usize>), AceError> {
    let n = bytes.len();
    let u16le = |p: usize| u16::from_le_bytes([bytes[p], bytes[p + 1]]) as usize;
    let u32le = |p: usize| {
        u32::from_le_bytes([bytes[p], bytes[p + 1], bytes[p + 2], bytes[p + 3]]) as usize
    };
    if n < 22 || &bytes[0..4] != b"PK\x03\x04" {
        return Err(AceError::Load("silence_latent.pt: не ZIP".into()));
    }
    // End-of-central-directory (PK\x05\x06), scanned from the tail.
    let lo = n.saturating_sub(65557);
    let mut eocd = None;
    let mut i = n - 22;
    while i >= lo {
        if &bytes[i..i + 4] == b"PK\x05\x06" {
            eocd = Some(i);
            break;
        }
        if i == 0 {
            break;
        }
        i -= 1;
    }
    let eocd = eocd.ok_or_else(|| AceError::Load("silence_latent.pt: нет EOCD".into()))?;
    let cd_count = u16le(eocd + 10);
    let cd_off = u32le(eocd + 16);
    let mut p = cd_off;
    let mut best: Option<(usize, usize)> = None; // (local_header_offset, uncompressed_size)
    for _ in 0..cd_count {
        if p + 46 > n || &bytes[p..p + 4] != b"PK\x01\x02" {
            break;
        }
        let method = u16le(p + 10);
        let uncomp = u32le(p + 24);
        let (fnlen, extlen, cmtlen) = (u16le(p + 28), u16le(p + 30), u16le(p + 32));
        let lho = u32le(p + 42);
        if method == 0 && best.map_or(true, |(_, s)| uncomp > s) {
            best = Some((lho, uncomp));
        }
        p += 46 + fnlen + extlen + cmtlen;
    }
    let (lho, usz) = best.ok_or_else(|| AceError::Load("silence_latent.pt: нет STORED-записи".into()))?;
    if lho + 30 > n || &bytes[lho..lho + 4] != b"PK\x03\x04" {
        return Err(AceError::Load("silence_latent.pt: битый local header".into()));
    }
    let data_off = lho + 30 + u16le(lho + 26) + u16le(lho + 28);
    if data_off + usz > n {
        return Err(AceError::Load("silence_latent.pt: данные за границей".into()));
    }
    let all: Vec<f32> = bytes[data_off..data_off + usz]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if all.is_empty() || all.len() % 64 != 0 {
        return Err(AceError::Load(format!("silence_latent.pt: numel {} не кратно 64", all.len())));
    }
    let t = all.len() / 64;
    Ok((all, vec![1, 64, t]))
}

pub fn load_silence_latent(dit_path: &Path, frames: usize, device: Device) -> Result<Tensor, AceError> {
    let (all, shape) = load_silence_raw(dit_path)?;
    let (t_full, ch, channel_major) = if shape.len() == 3 && shape[2] == 64 {
        (shape[1], 64usize, false)
    } else if shape.len() == 3 && shape[1] == 64 {
        (shape[2], 64usize, true)
    } else {
        return Err(AceError::Load(format!("silence_latent: неожиданная форма {shape:?}")));
    };
    let f = frames.min(t_full);
    let mut out = vec![0f32; frames * ch];
    for t in 0..f {
        for c in 0..ch {
            let src = if channel_major { c * t_full + t } else { t * ch + c };
            out[t * ch + c] = all[src];
        }
    }
    if f < frames {
        for t in f..frames {
            for c in 0..ch {
                out[t * ch + c] = out[(f - 1) * ch + c];
            }
        }
    }
    Ok(Tensor::from_vec(out, vec![1usize, frames, ch], device)?)
}

pub struct MusicPaths<'a> {
    pub lm: &'a Path,
    pub text_encoder: &'a Path,
    pub dit: &'a Path,
    pub vae: &'a Path,
}

/// Output loudness normalization mode (matches the old Rust output node).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NormMode {
    Off,
    #[default]
    Peak,
    Rms,
}

/// Extra generation controls exposed by the node/CLI on top of the sampler/codes
/// options: AR on/off (turbo skips the 5Hz LM), user metadata overrides
/// (bpm/keyscale/timesignature → DiT metas + CoT; `None` → "N/A" like Python),
/// and the output normalization mode.
#[derive(Debug, Clone, Default)]
pub struct GenExtras {
    pub use_ar: bool,
    pub bpm: Option<u32>,
    pub keyscale: Option<String>,
    pub timesig: Option<String>,
    pub norm_mode: NormMode,
}

impl GenExtras {
    /// Default full-pipeline behaviour: AR on, peak-normalized output.
    pub fn ar_on() -> Self {
        Self { use_ar: true, norm_mode: NormMode::Peak, ..Default::default() }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_music(
    paths: &MusicPaths,
    caption: &str,
    lyric: &str,
    duration_sec: u32,
    device: Device,
    compute: DType,
    dit_quant: DType,
    enc_quant: DType,
    opts: &SamplerOptions,
    codes_opts: &CodesGenOptions,
    use_cot: bool,
    edit: &EditOptions,
    extras: &GenExtras,
) -> Result<(Vec<f32>, u32, Tensor), AceError> {
    use crate::ar::ar_generate;
    let cfg = DitConfig::xl_base();
    let seed = codes_opts.seed;
    let enc_compute = compute;

    let auto = duration_sec == 0;
    let cot = use_cot || auto;

    let (codes, meta) = {
        let cap_sec = if auto { 600 } else { duration_sec as usize };
        // User metadata overrides (None -> "N/A" in the DiT metas, omitted from CoT).
        let base = Metadata {
            caption: caption.to_string(),
            duration: if auto { 30 } else { duration_sec },
            bpm: extras.bpm,
            keyscale: extras.keyscale.clone(),
            timesignature: extras.timesig.clone(),
            ..Metadata::default()
        };
        if extras.use_ar {
            let t_open = std::time::Instant::now();
            let lm = AceStepLm::open(paths.lm, device, enc_compute, enc_quant, cap_sec * 5 + 2048)?;
            eprintln!("[t] LM load: {:.1}s", t_open.elapsed().as_secs_f32());
            let lm_tok = AceTokenizer::from_bytes(&read_bundle_file(paths.lm, "tokenizer.json")?)?;
            let t_ar = std::time::Instant::now();
            let r = ar_generate(&lm, &lm_tok, caption, lyric, &base, codes_opts, cot)?;
            eprintln!("[t] AR generate ({} codes): {:.1}s", r.0.len(), t_ar.elapsed().as_secs_f32());
            r
        } else {
            // Turbo / AR off: no 5Hz LM codes — the DiT denoises from noise with a
            // silence source latent (frames derived from the explicit duration).
            eprintln!("[t] AR skipped (use_ar=false)");
            (Vec::new(), base)
        }
    };
    eprintln!(
        "[music] codes={} → ~{:.1}s (CoT duration={}s)",
        codes.len(),
        codes.len() as f32 / 5.0,
        meta.duration
    );

    let t2 = std::time::Instant::now();
    let (text_hidden, lyric_hidden) = {
        use crate::text_encoder::{build_lyric_prompt, build_text_prompt};
        let te = TextEncoder::open(paths.text_encoder, device, enc_compute, enc_quant, 4096)?;
        let tok = HfTokenizer::from_bytes(&read_bundle_file(paths.text_encoder, "tokenizer.json")?)
            .map_err(|e| AceError::Load(e.to_string()))?;
        let enc_ids = |s: &str| -> Result<Tensor, AceError> {
            let mut ids = tok.encode(s, false).map_err(|e| AceError::Other(e.to_string()))?.ids;
            if ids.is_empty() {
                ids.push(151643);
            }
            let n = ids.len();
            Ok(Tensor::from_vec(ids, vec![1usize, n], device)?)
        };
        let cap_prompt = build_text_prompt(
            caption,
            meta.duration,
            meta.bpm,
            meta.timesignature.as_deref(),
            meta.keyscale.as_deref(),
        );
        let lyr_prompt = build_lyric_prompt(lyric, meta.language.as_deref().unwrap_or("en"));
        let cap = te.caption_hidden(&enc_ids(&cap_prompt)?)?;
        let lyr = te.lyric_embed(&enc_ids(&lyr_prompt)?)?;
        (cap.to_dtype(DType::F32)?, lyr.to_dtype(DType::F32)?)
    };

    eprintln!("[t] text-enc (load+fwd): {:.1}s", t2.elapsed().as_secs_f32());

    let t3 = std::time::Instant::now();
    let dit_ck = CompLoader::open(paths.dit, None, device)?;
    let lm_hints = if codes.is_empty() {
        // No AR (turbo): silence source, frames from the explicit duration
        // (25 latent frames/sec). is_covers=False -> silence src (matches Python).
        let tf = (meta.duration as usize).max(1) * 25;
        load_silence_latent(paths.dit, tf, device)?
    } else {
        let fsq = Fsq::load(&dit_ck, "tokenizer.quantizer")?;
        let detok = Detokenizer::load(&dit_ck, &cfg)?;
        detok.forward(&fsq.get_output_from_indices(&codes)?)?
    };
    let enc_cond = {
        let cond = ConditionEncoder::load(&dit_ck, &cfg)?;
        let timbre_ref = load_silence_latent(paths.dit, cfg.timbre_fix_frame, device)?;
        cond.forward_full(&text_hidden, &lyric_hidden, &timbre_ref)?
    };
    let l = enc_cond.dims()[1];
    let null = dit_ck
        .f32("null_condition_emb")?
        .broadcast_as(vec![1usize, l, cfg.encoder_hidden_size])?
        .contiguous()?;

    let t_frames = lm_hints.dims()[1];
    let src_half = if matches!(edit.mode, EditMode::Extract) {
        let s = edit
            .src_latent
            .as_ref()
            .ok_or_else(|| AceError::Other("extract/cover: нужен src_latent на входе".into()))?;
        align_latent_len(s, t_frames, device)?
    } else {
        lm_hints.clone()
    };
    let chunk = Tensor::ones(vec![1usize, t_frames, 64], DType::F32, device)?;
    let context = Tensor::cat(&[&src_half, &chunk], 2)?;

    eprintln!("[t] cond (dit_ck load + fsq/detok/cond/silence): {:.1}s", t3.elapsed().as_secs_f32());

    let latent = {
        let t_dl = std::time::Instant::now();
        let dit = Dit::load(&dit_ck, &cfg, compute, dit_quant)?;
        eprintln!("[t] DiT load: {:.1}s", t_dl.elapsed().as_secs_f32());
        let x0 = {
            let base = Tensor::randn_seeded(vec![1usize, t_frames, 64], seed, Device::Cpu)?;
            let mixed = if matches!(edit.mode, EditMode::Retake) && edit.retake_variance > 0.0 {
                let v = (edit.retake_variance.clamp(0.0, 1.0)) * std::f32::consts::FRAC_PI_2;
                let rt = Tensor::randn_seeded(vec![1usize, t_frames, 64], edit.retake_seed, Device::Cpu)?;
                base.affine(v.cos(), 0.0)?.broadcast_add(&rt.affine(v.sin(), 0.0)?)?
            } else {
                base
            };
            mixed.to_device(device)?
        };
        let t_dn = std::time::Instant::now();
        let r = match edit.mode {
            EditMode::Repaint => {
                let src = edit
                    .src_latent
                    .as_ref()
                    .ok_or_else(|| AceError::Other("repaint: нужен src_latent на входе".into()))?;
                let src = align_latent_len(src, t_frames, device)?;
                let strength = edit.repaint_strength.clamp(0.0, 1.0);
                let inv = 1.0 - strength;
                let cutoff = ((inv * opts.steps as f32).round() as usize).min(opts.steps);
                let cf = (25.0 * inv).round() as usize;
                let (mask, soft) = build_repaint_masks(
                    t_frames,
                    edit.repaint_start_sec,
                    edit.repaint_end_sec,
                    cf,
                    device,
                )?;
                denoise_repaint(
                    &dit, &x0, &context, &enc_cond, Some(&null), opts, &src, &x0, &mask, &soft,
                    cutoff,
                )?
            }
            EditMode::Edit => {
                let src = edit
                    .src_latent
                    .as_ref()
                    .ok_or_else(|| AceError::Other("edit: нужен src_latent на входе".into()))?;
                let src = align_latent_len(src, t_frames, device)?;
                let t_start = edit.edit_n_max.clamp(0.0, 1.0);
                denoise_sdedit(&dit, &src, &x0, &context, &enc_cond, Some(&null), opts, t_start)?
            }
            _ => denoise(&dit, &x0, &context, &enc_cond, Some(&null), opts)?,
        };
        eprintln!("[t] denoise ({} steps, T={}): {:.1}s", opts.steps, t_frames, t_dn.elapsed().as_secs_f32());
        r
    };

    let t6 = std::time::Instant::now();
    let vae = AceStepVae::open(paths.vae, device)?;
    let latent_ncl = latent.transpose(1, 2)?.contiguous()?;
    let audio = vae.decode_tiled(&latent_ncl, 500, 32)?;
    let ch0 = audio.narrow(1, 0, 1)?.contiguous()?;
    let mut samples: Vec<f32> = ch0.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
    apply_norm(&mut samples, extras.norm_mode);
    eprintln!("[t] VAE (load+tiled decode): {:.1}s", t6.elapsed().as_secs_f32());
    Ok((samples, 48000, latent_ncl))
}

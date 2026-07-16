//! LTX-2.3 distilled txt→video пайплайн (одностадийный, без upscaler): шум →
//! 8-шаговая flow-match Euler денойз-петля (DiT velocity) → VAE decode → RGB.
//!
//! Сигмы distilled stage-1 (8 шагов). X0→Euler схлопывается до
//! `latent_next = latent + velocity·(σ_next − σ)`.

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::audio_vae::AudioVaeDecoder;
use crate::dit::{AvDit, VideoDit};
use crate::guider::{self, GuiderParams};
use crate::upscaler::Upsampler;
use crate::vae::VaeDecoder;
use crate::vocoder::VocoderWithBwe;
use crate::LtxError;

/// Аудио-латент-координаты `[1,Fa,2]` (секунды) для split-RoPE: token k →
/// start=max(4k−3,0)·0.01, end=(4k+1)·0.01. (AudioPatchifier causal,
/// downsample 4, hop 160, sr 16000 → 0.01с/мел-кадр.)
pub fn audio_coords(fa: usize) -> Vec<f64> {
    let mut p = vec![0f64; fa * 2];
    for k in 0..fa {
        p[k * 2] = ((4 * k) as f64 - 3.0).max(0.0) * 0.01;
        p[k * 2 + 1] = (4 * k + 1) as f64 * 0.01;
    }
    p
}

/// Distilled stage-2 сигмы (3 Euler-шага, re-noise при σ[0]).
pub const STAGE2_SIGMAS: [f64; 4] = [0.909375, 0.725, 0.421875, 0.0];

/// Distilled stage-1 сигмы (9 значений → 8 Euler-шагов).
pub const DISTILLED_SIGMAS: [f64; 9] =
    [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0];

/// Дефолтный NAG negative-prompt (подавление псевдо-субтитров/текста/вотермарок).
pub const DEFAULT_NAG_PROMPT: &str =
    "subtitles, text, captions, words on screen, watermark, overlay effects, still image, bad quality";

/// Дефолтные NAG-параметры (экстраполяция / бленд / L1-кламп).
pub const NAG_DEFAULT_SCALE: f32 = 11.0;
pub const NAG_DEFAULT_ALPHA: f32 = 0.25;
pub const NAG_DEFAULT_TAU: f32 = 2.5;

/// Поддерживаемые значения fps (rope-позиции по времени делятся на fps,
/// модель обучена на этих сетках).
pub const SUPPORTED_FPS: [f64; 4] = [24.0, 25.0, 48.0, 50.0];

/// Латентная сетка по пикселям: H=32·hp, W=32·wp.
pub fn latent_grid(width: usize, height: usize) -> (usize, usize) {
    (height.div_ceil(32).max(1), width.div_ceil(32).max(1))
}

/// Латент-кадры из длительности: round к ближайшему 8·(fp−1)+1, чтобы
/// длительность держалась точнее.
pub fn frames_for_duration(secs: f64, fps: f64) -> usize {
    ((((secs * fps - 1.0) / 8.0).round() as usize) + 1).max(1)
}

/// Латент-кадры из явного числа пиксель-кадров (floor к сетке, back-compat).
pub fn fp_for_frames(frames: usize) -> usize {
    ((frames.saturating_sub(1)) / 8 + 1).max(1)
}

/// Число пиксель-кадров видео из латент-кадров: F=8·(F'−1)+1.
pub fn out_frame_count(fp: usize) -> usize {
    8 * (fp.saturating_sub(1)) + 1
}

/// Сетка stage1 двухстадийного пайплайна — половина целевой (upscaler ×2
/// восстанавливает).
pub fn stage1_grid(hp: usize, wp: usize) -> (usize, usize) {
    (hp.div_ceil(2).max(1), wp.div_ceil(2).max(1))
}

/// Прогресс одного Euler-шага денойз-петли (для UI поверх библиотеки).
#[derive(Clone, Copy, Debug)]
pub struct DenoiseProgress {
    pub step: usize,
    pub total: usize,
    pub sigma: f64,
}

/// Хуки денойз-петли: колбэк прогресса + флаг отмены. [`DenoiseHooks::none`]
/// — поведение как раньше (без колбэков, без отмены).
#[derive(Default)]
pub struct DenoiseHooks<'a> {
    pub progress: Option<&'a (dyn Fn(DenoiseProgress) + Sync)>,
    pub cancel: Option<&'a std::sync::atomic::AtomicBool>,
}

impl<'a> DenoiseHooks<'a> {
    pub fn none() -> Self {
        Self::default()
    }

    fn cancelled(&self) -> bool {
        self.cancel
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed))
            .unwrap_or(false)
    }

    fn emit(&self, step: usize, total: usize, sigma: f64) {
        if let Some(p) = self.progress {
            p(DenoiseProgress { step, total, sigma });
        }
    }
}

fn noise_tensor(shape: Vec<usize>, seed: Option<u64>) -> Result<Tensor, LtxError> {
    match seed {
        Some(s) => Tensor::randn_seeded(shape, s, Device::Cpu).map_err(LtxError::from),
        None => Tensor::randn(shape, Device::Cpu).map_err(LtxError::from),
    }
}

/// Pixel-координаты токенов латентной сетки `F'×H'×W'` (f-major), формат
/// `[3, T, 2]` (start,end) flat → `[(d*T+token)*2 + bound]`. Масштаб VAE
/// (время 8, H/W 32), causal_fix первого кадра по времени.
pub fn pixel_coords(fp: usize, hp: usize, wp: usize, fps: f64) -> Vec<f64> {
    let (st, sh, sw) = (8f64, 32f64, 32f64);
    let t = fp * hp * wp;
    let mut p = vec![0f64; 3 * t * 2];
    for f in 0..fp {
        for h in 0..hp {
            for w in 0..wp {
                let tok = (f * hp + h) * wp + w;
                // d=0 время: causal_fix (idx*scale + 1 - scale).max(0), затем ÷fps
                // (как python VideoLatentTools: positions[:,0,...] /= fps — БЕЗ
                // этого деления rope-позиции по времени в fps× больше → модель вне
                // распределения → вырожденная динамика, мало движения, артефакты).
                let t0 = ((f as f64) * st + 1.0 - st).max(0.0) / fps;
                let t1 = ((f as f64 + 1.0) * st + 1.0 - st).max(0.0) / fps;
                p[(0 * t + tok) * 2] = t0;
                p[(0 * t + tok) * 2 + 1] = t1;
                // d=1 высота, d=2 ширина (без causal_fix)
                p[(1 * t + tok) * 2] = h as f64 * sh;
                p[(1 * t + tok) * 2 + 1] = (h as f64 + 1.0) * sh;
                p[(2 * t + tok) * 2] = w as f64 * sw;
                p[(2 * t + tok) * 2 + 1] = (w as f64 + 1.0) * sw;
            }
        }
    }
    p
}

/// Сгенерировать видео-латент через distilled-петлю и декодировать в RGB.
/// `video_encoding` `[1,T_txt,4096]` (выход текст-кондишена). `fp/hp/wp` —
/// размеры латентной сетки. Возвращает RGB `[1,3,F,H,W]`, F=8·(F'−1)+1.
#[allow(clippy::too_many_arguments)]
pub fn generate_video(
    dit: &VideoDit,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let t = fp * hp * wp;
    let positions = pixel_coords(fp, hp, wp, fps);
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;

    // sigma[0]=1.0 → начальный латент = чистый шум (randn только на CPU → переносим).
    let noise = Tensor::randn(vec![1usize, 128, fp, hp, wp], Device::Cpu)?
        .to_device(device)?
        .to_dtype(DType::BF16)?;
    // patchify (patch_size=1): [1,128,F',H',W'] → [1,128,T] → [1,T,128]
    let mut tokens = noise.reshape(vec![1, 128, t])?.transpose(1, 2)?.contiguous()?;

    let prof = crate::runtime::ltx_prof();
    let ord = if let Device::Cuda(o) = device { o } else { 0 };
    let sync = || { let _ = synaptix_core::device::cuda::synchronize(ord); };

    let max_steps = DISTILLED_SIGMAS.len() - 1;
    let t_loop = std::time::Instant::now();
    for i in 0..max_steps {
        let sigma = DISTILLED_SIGMAS[i];
        let sigma_next = DISTILLED_SIGMAS[i + 1];
        let ts = vec![sigma as f32; t];
        let t_step = std::time::Instant::now();
        let velocity = dit.forward(&tokens, &ts, sigma as f32, &positions, &ctx)?; // [1,T,128]
        // Euler: tokens += velocity * (σ_next − σ)
        let step = velocity.mul_scalar((sigma_next - sigma) as f32)?;
        tokens = tokens.add(&step)?;
        if prof { sync(); eprintln!("[LTX_PROF] step {i} ({t} ток): {:.2}s", t_step.elapsed().as_secs_f32()); }
    }
    if prof { sync(); eprintln!("[LTX_PROF] denoise-петля ({} шагов): {:.2}s", DISTILLED_SIGMAS.len() - 1, t_loop.elapsed().as_secs_f32()); }

    // unpatchify: [1,T,128] → [1,128,F',H',W']
    let latent = tokens.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?;
    let t_vae = std::time::Instant::now();
    let rgb = vae.decode(&latent)?;
    if prof { sync(); eprintln!("[LTX_PROF] VAE decode: {:.2}s", t_vae.elapsed().as_secs_f32()); }
    Ok(rgb)
}

/// Two-stage HQ: stage1 (8 шагов) → нативный latent-upscaler ×2 → опц. stage2
/// re-noise+refine (3 шага) → VAE decode. Возвращает RGB `[1,3,F,H·64,W·64]`
/// (upscaler ×2 latent + VAE ×32). `stage2=false` → апскейл без рефайна (быстрее).
#[allow(clippy::too_many_arguments)]
pub fn generate_two_stage(
    dit: &VideoDit,
    upscaler: &Upsampler,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    stage2: bool,
    device: Device,
) -> Result<Tensor, LtxError> {
    let latent = generate_two_stage_latent(dit, upscaler, video_encoding, fp, hp, wp, fps, stage2, device)?;
    vae.decode(&latent)
}

/// Как [`generate_two_stage`], но возвращает ЛАТЕНТ `[1,128,F',H'·2,W'·2]` без VAE
/// decode — чтобы вызывающий мог ДРОПНУТЬ DiT (освободить VRAM) перед тяжёлым
/// decode (критично для HD: резидентный mxfp8-DiT ~20GB + VAE-активации > 24GB).
#[allow(clippy::too_many_arguments)]
pub fn generate_two_stage_latent(
    dit: &VideoDit,
    upscaler: &Upsampler,
    video_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    stage2: bool,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let l1 = denoise(dit, &ctx, fp, hp, wp, &DISTILLED_SIGMAS, None, fps, device, None, &DenoiseHooks::none())?;
    let l2 = upscaler.upsample(&l1)?.to_dtype(DType::BF16)?; // [1,128,F',H'·2,W'·2]
    let (hp2, wp2) = (hp * 2, wp * 2);
    if stage2 {
        denoise(dit, &ctx, fp, hp2, wp2, &STAGE2_SIGMAS, Some(&l2), fps, device, None, &DenoiseHooks::none())
    } else {
        l2.to_dtype(DType::F32).map_err(LtxError::from)
    }
}

/// Денойз-петля: если `init` задан — re-noise при σ[0] (GaussianNoiser:
/// noise·σ₀ + clean·(1−σ₀)); иначе чистый шум. → латент `[1,128,F',H',W']`.
/// Публична для постадийной сборки two-stage с РАЗНЫМИ dit (per-stage LoRA,
/// официальный HQ-паттерн strength stage1=0.25/stage2=0.5) и последовательным
/// жизненным циклом dit (mxfp8: два резидентных не влезают).
#[allow(clippy::too_many_arguments)]
pub fn denoise(
    dit: &VideoDit,
    ctx: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    init: Option<&Tensor>,
    fps: f64,
    device: Device,
    seed: Option<u64>,
    hooks: &DenoiseHooks,
) -> Result<Tensor, LtxError> {
    let t = fp * hp * wp;
    let positions = pixel_coords(fp, hp, wp, fps);
    let noise = noise_tensor(vec![1usize, 128, fp, hp, wp], seed)?
        .to_device(device)?.to_dtype(DType::BF16)?;
    // начальный латент
    let s0 = sigmas[0];
    let latent0 = match init {
        Some(clean) => {
            let clean = clean.to_device(device)?.to_dtype(DType::BF16)?;
            noise.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?
        }
        None => noise,
    };
    let mut tokens = latent0.reshape(vec![1, 128, t])?.transpose(1, 2)?.contiguous()?;
    let total = sigmas.len() - 1;
    for i in 0..total {
        if hooks.cancelled() {
            return Err(LtxError::Cancelled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let ts = vec![sigma as f32; t];
        let vel = dit.forward(&tokens, &ts, sigma as f32, &positions, ctx)?;
        tokens = tokens.add(&vel.mul_scalar((sigma_next - sigma) as f32)?)?;
        hooks.emit(i + 1, total, sigma_next);
    }
    Ok(tokens.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?
        .to_dtype(DType::F32)?)
}

/// `LTX2Scheduler`: сигма-расписание с token-зависимым сдвигом + растяжением к
/// terminal. Bit-faithful к `schedulers.py::LTX2Scheduler.execute` (max_shift 2.05,
/// base_shift 0.95, stretch=true, terminal=0.1, power=1). `tokens` = число латентных
/// токенов (F'·H'·W'); анкоры x1=1024, x2=4096. Возвращает `steps+1` сигм (1.0→0.0).
pub fn ltx2_sigmas(steps: usize, tokens: usize) -> Vec<f64> {
    let (base_shift, max_shift, x1, x2) = (0.95f64, 2.05f64, 1024f64, 4096f64);
    let mm = (max_shift - base_shift) / (x2 - x1);
    let b = base_shift - mm * x1;
    let esh = (tokens as f64 * mm + b).exp();
    let n = steps + 1;
    // linspace(1.0, 0.0, n) → токен-сдвиг exp(shift)/(exp(shift)+(1/s−1))
    let mut sig: Vec<f64> = (0..n)
        .map(|i| {
            let s = 1.0 - i as f64 / (n - 1) as f64;
            if s != 0.0 { esh / (esh + (1.0 / s - 1.0)) } else { 0.0 }
        })
        .collect();
    // stretch: последний ненулевой → terminal 0.1
    let terminal = 0.1;
    let last_nz = sig.iter().rposition(|&x| x != 0.0).unwrap();
    let scale = (1.0 - sig[last_nz]) / (1.0 - terminal);
    for s in sig.iter_mut() {
        if *s != 0.0 {
            *s = 1.0 - (1.0 - *s) / scale;
        }
    }
    sig
}

/// Guided денойз-петля видео (multimodal guidance: CFG + STG + rescale; isolated
/// modality только для A/V). На каждом шаге считает cond + (опц.) uncond_text +
/// (опц.) uncond_perturbed проходы DiT, комбинирует через [`guider::calculate`] и
/// делает Euler-шаг. `neg_ctx` — negative-context (CFG). Для distilled (без
/// guidance) используйте [`denoise`]. → латент `[1,128,F',H',W']`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_video_guided(
    dit: &VideoDit,
    ctx: &Tensor,
    neg_ctx: &Tensor,
    gp: &GuiderParams,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    init: Option<&Tensor>,
    fps: f64,
    device: Device,
    seed: Option<u64>,
    hooks: &DenoiseHooks,
) -> Result<Tensor, LtxError> {
    let t = fp * hp * wp;
    let positions = pixel_coords(fp, hp, wp, fps);
    let noise = noise_tensor(vec![1usize, 128, fp, hp, wp], seed)?
        .to_device(device)?.to_dtype(DType::BF16)?;
    let s0 = sigmas[0];
    let latent0 = match init {
        Some(clean) => {
            let clean = clean.to_device(device)?.to_dtype(DType::BF16)?;
            noise.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?
        }
        None => noise,
    };
    let mut tokens = latent0.reshape(vec![1, 128, t])?.transpose(1, 2)?.contiguous()?;
    let total = sigmas.len() - 1;
    for i in 0..total {
        if hooks.cancelled() {
            return Err(LtxError::Cancelled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let ts = vec![sigma as f32; t];
        let vel = if gp.should_skip_step(i) {
            dit.forward(&tokens, &ts, sigma as f32, &positions, ctx)?
        } else {
            // потоки guidance: cond + [uncond_text] + [perturbed] — ОДИН стрим-свип
            // блоков на все потоки (forward_multi): offload-стриминг 46GB ×1, не ×N.
            let (du, dp) = (gp.do_uncond(), gp.do_perturbed());
            let mut lat: Vec<&Tensor> = vec![&tokens];
            let mut tss: Vec<Vec<f32>> = vec![ts.clone()];
            let mut sigs: Vec<f32> = vec![sigma as f32];
            let mut ctxs: Vec<&Tensor> = vec![ctx];
            let mut pert: Vec<bool> = vec![false];
            if du { lat.push(&tokens); tss.push(ts.clone()); sigs.push(sigma as f32); ctxs.push(neg_ctx); pert.push(false); }
            if dp { lat.push(&tokens); tss.push(ts.clone()); sigs.push(sigma as f32); ctxs.push(ctx); pert.push(true); }
            let outs = dit.forward_multi(&lat, &tss, &sigs, &positions, &ctxs, &pert, &gp.stg_blocks)?;
            let mut idx = 1;
            let ut = if du { let r = idx; idx += 1; Some(&outs[r]) } else { None };
            let up = if dp { Some(&outs[idx]) } else { None };
            guider::calculate(gp, &outs[0], ut, up, None)?
        };
        tokens = tokens.add(&vel.mul_scalar((sigma_next - sigma) as f32)?)?;
    }
    Ok(tokens.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?
        .to_dtype(DType::F32)?)
}

/// Денойз видео с conditioning (image/keyframe replace-latent). `cond` =
/// `(start_token, tokens [1,n,128], strength)` — энкоженный кадр в позициях
/// `start..start+n`, strength∈[0,1] (1=полная замена). Bit-faithful к
/// `VideoConditionByLatentIndex` + `post_process_latent`/`timesteps_from_mask`:
/// per-token timesteps = mask·sigma (mask=1−strength у conditioned), euler с
/// per-token dt, после шага blend `tokens·mask + clean·(1−mask)`. → латент.
#[allow(clippy::too_many_arguments)]
pub fn denoise_video_conditioned(
    dit: &VideoDit,
    ctx: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    conds: &[(usize, Tensor, f32)],
    init: Option<&Tensor>,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let t = fp * hp * wp;
    let positions = pixel_coords(fp, hp, wp, fps);
    // mask по токенам (1=денойзим; 1−strength у conditioned)
    let mut mask_v = vec![1f32; t];
    for (start, toks, strength) in conds {
        let n = toks.dims()[1];
        for k in *start..(*start + n).min(t) {
            mask_v[k] = 1.0 - *strength;
        }
    }
    let mask = Tensor::from_vec(mask_v.clone(), vec![1, t, 1], device)?.to_dtype(DType::BF16)?;
    let inv_mask = mask.affine(-1.0, 1.0)?; // 1−mask
    // начальный латент: шум, либо (stage2) re-noise noise·σ0 + init·(1−σ0). Затем
    // замена conditioned-позиций на image-токены; clean = то же.
    let noise = Tensor::randn(vec![1usize, 128, fp, hp, wp], Device::Cpu)?
        .to_device(device)?.to_dtype(DType::BF16)?;
    let latent0 = match init {
        Some(clean) => {
            let s0 = sigmas[0];
            let clean = clean.to_device(device)?.to_dtype(DType::BF16)?;
            noise.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?
        }
        None => noise,
    };
    let mut tokens = latent0.reshape(vec![1, 128, t])?.transpose(1, 2)?.contiguous()?;
    let mut clean = tokens.clone();
    for (start, toks, _) in conds {
        let toks = toks.to_device(device)?.to_dtype(DType::BF16)?;
        let n = toks.dims()[1];
        // заменить строки [start:start+n] токенами: cat(pre, toks, post)
        let pre = if *start > 0 { Some(tokens.narrow(1, 0, *start)?.contiguous()?) } else { None };
        let post = if start + n < t { Some(tokens.narrow(1, start + n, t - start - n)?.contiguous()?) } else { None };
        let mut parts: Vec<Tensor> = Vec::new();
        if let Some(p) = &pre { parts.push(p.clone()); }
        parts.push(toks.clone());
        if let Some(p) = &post { parts.push(p.clone()); }
        let refs: Vec<&Tensor> = parts.iter().collect();
        tokens = Tensor::cat(&refs, 1)?.contiguous()?;
        clean = tokens.clone();
    }
    for i in 0..sigmas.len() - 1 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // per-token timesteps = mask·sigma
        let ts: Vec<f32> = mask_v.iter().map(|&m| m * sigma as f32).collect();
        let vel = dit.forward(&tokens, &ts, sigma as f32, &positions, ctx)?;
        // euler с per-token dt = mask·(σ_next−σ)
        let dt = mask.mul_scalar((sigma_next - sigma) as f32)?; // [1,t,1]
        tokens = tokens.add(&vel.broadcast_mul(&dt)?)?;
        // blend: tokens·mask + clean·(1−mask)
        tokens = tokens.broadcast_mul(&mask)?.add(&clean.broadcast_mul(&inv_mask)?)?;
    }
    Ok(tokens.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?
        .to_dtype(DType::F32)?)
}

/// Латент кадра `[1,128,1,H',W']` → conditioning-токены `[1,H'·W',128]` (patchify
/// patch_size=1: канал-в-последний). Для `denoise_video_conditioned` (image cond).
pub fn frame_latent_to_tokens(latent: &Tensor) -> Result<Tensor, LtxError> {
    let (c, h, w) = (latent.dims()[1], latent.dims()[3], latent.dims()[4]);
    Ok(latent.reshape(vec![1, c, h * w])?.transpose(1, 2)?.contiguous()?)
}

/// Pixel-позиции keyframe-токенов (1 латент-кадр `hp×wp` на пиксель-кадре
/// `frame_idx`), формат `[3, hp·wp, 2]` flat. Bit-faithful к
/// `VideoConditionByKeyframeIndex`: time = `[frame_idx/fps, (frame_idx+1)/fps]`
/// (single pixel frame), spatial как у видео-кадра (h·32, w·32). causal_fix НЕ
/// применяется для frame_idx>0.
pub fn keyframe_positions(hp: usize, wp: usize, frame_idx: usize, fps: f64) -> Vec<f64> {
    let (sh, sw) = (32f64, 32f64);
    let t = hp * wp;
    let mut p = vec![0f64; 3 * t * 2];
    let (t0, t1) = (frame_idx as f64 / fps, (frame_idx as f64 + 1.0) / fps);
    for h in 0..hp {
        for w in 0..wp {
            let tok = h * wp + w;
            p[(0 * t + tok) * 2] = t0;
            p[(0 * t + tok) * 2 + 1] = t1;
            p[(1 * t + tok) * 2] = h as f64 * sh;
            p[(1 * t + tok) * 2 + 1] = (h as f64 + 1.0) * sh;
            p[(2 * t + tok) * 2] = w as f64 * sw;
            p[(2 * t + tok) * 2 + 1] = (w as f64 + 1.0) * sw;
        }
    }
    p
}

/// Объединить два набора pixel-координат `[3,Ta,2]`+`[3,Tb,2]` (d-major flat) в
/// `[3,Ta+Tb,2]` (токены b идут после a). Для append-conditioning (keyframe).
fn cat_positions(a: &[f64], ta: usize, b: &[f64], tb: usize) -> Vec<f64> {
    let t = ta + tb;
    let mut p = vec![0f64; 3 * t * 2];
    for d in 0..3 {
        for tok in 0..ta {
            for bd in 0..2 {
                p[(d * t + tok) * 2 + bd] = a[(d * ta + tok) * 2 + bd];
            }
        }
        for tok in 0..tb {
            for bd in 0..2 {
                p[(d * t + ta + tok) * 2 + bd] = b[(d * tb + tok) * 2 + bd];
            }
        }
    }
    p
}

/// Keyframe-conditioned денойз (append): keyframe-токены `kf` `[1,Tk,128]` (на
/// пиксель-кадре `frame_idx`, позиции [`keyframe_positions`]) ДОБАВЛЯЮТСЯ к видео-
/// последовательности (не replace, как `VideoConditionByKeyframeIndex`); mask
/// keyframe=`1−strength` фиксирует их, видео денойзится. После — извлекаем первые
/// `Tv` (видео) токенов → латент `[1,128,F',H',W']`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_video_keyframe(
    dit: &VideoDit,
    ctx: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    kf: &Tensor,
    frame_idx: usize,
    strength: f32,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let pos_k = keyframe_positions(hp, wp, frame_idx, fps);
    denoise_video_append(dit, ctx, fp, hp, wp, sigmas, kf, &pos_k, strength, fps, device)
}

/// Append-conditioned денойз (общее ядро keyframe / IC-LoRA reference): к видео-
/// последовательности добавляются токены `app` `[1,Ta,128]` со своими позициями
/// `app_positions` (формат `[3,Ta,2]` flat); mask append=`1−strength` фиксирует их,
/// видео денойзится. После — извлекаем видео-токены `[1,128,F',H',W']`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_video_append(
    dit: &VideoDit,
    ctx: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    app: &Tensor,
    app_positions: &[f64],
    strength: f32,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let tv = fp * hp * wp;
    let ta = app.dims()[1];
    let t = tv + ta;
    let pos_v = pixel_coords(fp, hp, wp, fps);
    let positions = cat_positions(&pos_v, tv, app_positions, ta);
    let mut mask_v = vec![1f32; tv];
    mask_v.extend(std::iter::repeat(1.0 - strength).take(ta));
    let mask = Tensor::from_vec(mask_v.clone(), vec![1, t, 1], device)?.to_dtype(DType::BF16)?;
    let inv_mask = mask.affine(-1.0, 1.0)?;
    let noise = Tensor::randn(vec![1usize, 128, fp, hp, wp], Device::Cpu)?
        .to_device(device)?.to_dtype(DType::BF16)?;
    let v_tok = noise.reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?;
    let app = app.to_device(device)?.to_dtype(DType::BF16)?;
    let mut tokens = Tensor::cat(&[&v_tok, &app], 1)?.contiguous()?;
    let clean = tokens.clone();
    for i in 0..sigmas.len() - 1 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let ts: Vec<f32> = mask_v.iter().map(|&m| m * sigma as f32).collect();
        let vel = dit.forward(&tokens, &ts, sigma as f32, &positions, ctx)?;
        let dt = mask.mul_scalar((sigma_next - sigma) as f32)?;
        tokens = tokens.add(&vel.broadcast_mul(&dt)?)?;
        tokens = tokens.broadcast_mul(&mask)?.add(&clean.broadcast_mul(&inv_mask)?)?;
    }
    let v = tokens.narrow(1, 0, tv)?.contiguous()?;
    Ok(v.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?.to_dtype(DType::F32)?)
}

/// Pixel-позиции reference-видео (IC-LoRA): `[fp×hp×wp]`-сетка, spatial-оси
/// масштабируются на `downscale` (reference в downscale× меньшем разрешении). Время
/// как [`pixel_coords`]. Bit-faithful к `VideoConditionByReferenceLatent`.
pub fn ref_video_positions(fp: usize, hp: usize, wp: usize, fps: f64, downscale: usize) -> Vec<f64> {
    let mut p = pixel_coords(fp, hp, wp, fps);
    if downscale != 1 {
        let t = fp * hp * wp;
        for d in [1usize, 2] {
            for tok in 0..t {
                p[(d * t + tok) * 2] *= downscale as f64;
                p[(d * t + tok) * 2 + 1] *= downscale as f64;
            }
        }
    }
    p
}

/// IC-LoRA video→video: reference-видео `ref_frames` `[1,3,F,Hr,Wr]` ([−1,1],
/// разрешение target/downscale) → VAE encode → append как control-сигнал (IC-LoRA
/// адаптер уже мерджнут в `dit`) → денойз → decode. `strength` (1=reference clean).
/// fp/hp/wp — целевая сетка; reference-сетка = fp×(hp/downscale)×(wp/downscale).
#[allow(clippy::too_many_arguments)]
pub fn generate_ic_lora_video(
    dit: &VideoDit,
    encoder: &crate::vae::VaeEncoder,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    ref_frames: &Tensor,
    downscale: usize,
    strength: f32,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let ref_latent = encoder.encode(ref_frames)?; // [1,128,fpr,hpr,wpr]
    let (fpr, hpr, wpr) = (ref_latent.dims()[2], ref_latent.dims()[3], ref_latent.dims()[4]);
    let tokens = ref_latent.reshape(vec![1, 128, fpr * hpr * wpr])?.transpose(1, 2)?.contiguous()?;
    let positions = ref_video_positions(fpr, hpr, wpr, fps, downscale);
    let latent = denoise_video_append(dit, &ctx, fp, hp, wp, &DISTILLED_SIGMAS, &tokens, &positions, strength, fps, device)?;
    vae.decode(&latent)
}

/// Image→video (distilled, одна стадия): кадр `image` `[1,3,1,H'·32,W'·32]` ([−1,1])
/// энкодится VAE-энкодером → conditioning первого латент-кадра (`strength`, 1=полная
/// замена) → conditioned distilled-денойз → VAE decode → RGB `[1,3,F,H,W]`. `video_encoding`
/// — текст-кондишен (промпт описывает движение/сцену вокруг изображения).
#[allow(clippy::too_many_arguments)]
pub fn generate_image_to_video(
    dit: &VideoDit,
    encoder: &crate::vae::VaeEncoder,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    image: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    strength: f32,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let img_latent = encoder.encode(image)?; // [1,128,1,hp,wp]
    let tokens = frame_latent_to_tokens(&img_latent)?; // [1, hp·wp, 128]
    let conds = vec![(0usize, tokens, strength)];
    let latent = denoise_video_conditioned(dit, &ctx, fp, hp, wp, &DISTILLED_SIGMAS, &conds, None, fps, device)?;
    vae.decode(&latent)
}

/// Keyframe→video (distilled, одна стадия): кадр `image` на пиксель-кадре
/// `frame_idx` (>0) добавляется как keyframe-conditioning (append) → денойз →
/// decode. frame_idx=0 эквивалентен replace ([`generate_image_to_video`]).
#[allow(clippy::too_many_arguments)]
pub fn generate_keyframe_to_video(
    dit: &VideoDit,
    encoder: &crate::vae::VaeEncoder,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    image: &Tensor,
    frame_idx: usize,
    fp: usize,
    hp: usize,
    wp: usize,
    strength: f32,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let img_latent = encoder.encode(image)?;
    let tokens = frame_latent_to_tokens(&img_latent)?;
    let latent = denoise_video_keyframe(dit, &ctx, fp, hp, wp, &DISTILLED_SIGMAS, &tokens, frame_idx, strength, fps, device)?;
    vae.decode(&latent)
}

/// denoise-маска retake: для каждого латент-кадра `f` (его time-границы как в
/// [`pixel_coords`] d=0) mask=1 если `[t0,t1]` пересекает `[start,end]` (регенерим),
/// иначе 0 (frozen). Bit-faithful к `TemporalRegionMask` (in_region = t_end>start &
/// t_start<end). → вектор `[fp·hp·wp]`.
pub fn retake_mask(fp: usize, hp: usize, wp: usize, start_time: f64, end_time: f64, fps: f64) -> Vec<f32> {
    let t = fp * hp * wp;
    let mut m = vec![0f32; t];
    for f in 0..fp {
        let t0 = ((f as f64) * 8.0 + 1.0 - 8.0).max(0.0) / fps;
        let t1 = ((f as f64 + 1.0) * 8.0 + 1.0 - 8.0).max(0.0) / fps;
        if t1 > start_time && t0 < end_time {
            for tok in (f * hp * wp)..((f + 1) * hp * wp) {
                m[tok] = 1.0;
            }
        }
    }
    m
}

/// Retake: перегенерировать временно́й регион `[start,end]` секунд исходного
/// видео-латента `source` `[1,128,F',H',W']`, сохранив остальное. denoise_mask=1
/// внутри региона (re-noise+денойз), 0 снаружи (frozen к source). Bit-faithful к
/// `TemporalRegionMask` + `post_process_latent`. → латент `[1,128,F',H',W']`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_video_retake(
    dit: &VideoDit,
    ctx: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    source: &Tensor,
    start_time: f64,
    end_time: f64,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let t = fp * hp * wp;
    let positions = pixel_coords(fp, hp, wp, fps);
    let mask_v = retake_mask(fp, hp, wp, start_time, end_time, fps);
    let mask = Tensor::from_vec(mask_v.clone(), vec![1, t, 1], device)?.to_dtype(DType::BF16)?;
    let inv_mask = mask.affine(-1.0, 1.0)?;
    // clean = ИСХОДНЫЙ латент (frozen-цель для blend), не re-noised
    let clean = source.to_device(device)?.to_dtype(DType::BF16)?
        .reshape(vec![1, 128, t])?.transpose(1, 2)?.contiguous()?;
    let noise = Tensor::randn(vec![1usize, 128, fp, hp, wp], Device::Cpu)?
        .to_device(device)?.to_dtype(DType::BF16)?
        .reshape(vec![1, 128, t])?.transpose(1, 2)?.contiguous()?;
    // re-noise: latent0 = noise·σ0 + clean·(1−σ0)
    let s0 = sigmas[0];
    let mut tokens = noise.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?;
    for i in 0..sigmas.len() - 1 {
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let ts: Vec<f32> = mask_v.iter().map(|&m| m * sigma as f32).collect();
        let vel = dit.forward(&tokens, &ts, sigma as f32, &positions, ctx)?;
        let dt = mask.mul_scalar((sigma_next - sigma) as f32)?;
        tokens = tokens.add(&vel.broadcast_mul(&dt)?)?;
        tokens = tokens.broadcast_mul(&mask)?.add(&clean.broadcast_mul(&inv_mask)?)?;
    }
    Ok(tokens.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?.to_dtype(DType::F32)?)
}

/// Joint A/V retake: видео-поток ре-генерируется в temporal-регионе `[start,end]`
/// (re-noise source с retake-маской, frozen снаружи — как [`denoise_video_retake`]),
/// аудио-поток генерируется заново под промпт. `source` — видео-латент
/// `[1,128,F',H',W']` исходного видео (VAE-encode). Возвращает
/// `(видео-латент f32, аудио-токены)`.
#[allow(clippy::too_many_arguments)]
pub fn denoise_av_retake(
    dit: &AvDit,
    video_encoding: &Tensor,
    audio_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    source: &Tensor,
    start_time: f64,
    end_time: f64,
    fps: f64,
    device: Device,
    seed: Option<u64>,
    hooks: &DenoiseHooks,
) -> Result<(Tensor, Tensor), LtxError> {
    let tv = fp * hp * wp;
    let ddt = dit.compute_dtype();
    let v_ctx = video_encoding.to_device(device)?.to_dtype(ddt)?;
    let a_ctx = audio_encoding.to_device(device)?.to_dtype(ddt)?;
    let fa = audio_token_count(fp, fps);
    let v_pos = pixel_coords(fp, hp, wp, fps);
    let a_pos = audio_coords(fa);
    let s0 = sigmas[0];

    // видео: re-noise source, retake-mask (1=регенерим, 0=frozen к source)
    let v_mask_v = retake_mask(fp, hp, wp, start_time, end_time, fps);
    let v_mask = Tensor::from_vec(v_mask_v.clone(), vec![1, tv, 1], device)?.to_dtype(ddt)?;
    let v_inv = v_mask.affine(-1.0, 1.0)?;
    let v_clean = source.to_device(device)?.to_dtype(ddt)?
        .reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?;
    let v_noise = noise_tensor(vec![1usize, 128, fp, hp, wp], seed)?.to_device(device)?.to_dtype(ddt)?
        .reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?;
    let mut v_tok = v_noise.mul_scalar(s0 as f32)?.add(&v_clean.mul_scalar((1.0 - s0) as f32)?)?;

    // аудио: обычный денойз с нуля (свежий звук под видео)
    let a_noise = noise_tensor(vec![1usize, 8, fa, 16], seed)?.to_device(device)?.to_dtype(ddt)?;
    let mut a_tok = a_noise.transpose(1, 2)?.contiguous()?.reshape(vec![1, fa, 128])?;

    for i in 0..sigmas.len() - 1 {
        if hooks.cancelled() {
            return Err(LtxError::Cancelled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let v_ts: Vec<f32> = v_mask_v.iter().map(|&m| m * sigma as f32).collect();
        let a_ts = vec![sigma as f32; fa];
        let (v_vel, a_vel) = dit.forward(
            &v_tok, &v_ts, sigma as f32, &v_pos, &v_ctx,
            &a_tok, &a_ts, sigma as f32, &a_pos, &a_ctx,
            None,
        )?;
        let dt = (sigma_next - sigma) as f32;
        v_tok = v_tok.add(&v_vel.broadcast_mul(&v_mask.mul_scalar(dt)?)?)?;
        v_tok = v_tok.broadcast_mul(&v_mask)?.add(&v_clean.broadcast_mul(&v_inv)?)?;
        a_tok = a_tok.add(&a_vel.mul_scalar(dt)?)?;
        hooks.emit(i + 1, sigmas.len() - 1, sigma_next);
    }
    let v_latent = v_tok.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?.to_dtype(DType::F32)?;
    Ok((v_latent, a_tok))
}

/// Retake→video (distilled): кадры `frames` `[1,3,F,H,W]` ([−1,1]) исходного видео
/// → VAE encode → retake региона `[start,end]` → VAE decode → RGB.
#[allow(clippy::too_many_arguments)]
pub fn generate_retake_video(
    dit: &VideoDit,
    encoder: &crate::vae::VaeEncoder,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    frames: &Tensor,
    start_time: f64,
    end_time: f64,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let source = encoder.encode(frames)?; // [1,128,fp,hp,wp]
    let latent = denoise_video_retake(dit, &ctx, fp, hp, wp, &DISTILLED_SIGMAS, &source, start_time, end_time, fps, device)?;
    vae.decode(&latent)
}

/// HQ image→video (distilled two-stage): conditioned stage1 (half-res, image на
/// кадре 0) → spatial-upscaler ×2 → conditioned stage2-refine (full-res, image
/// энкодится на stage2-разрешении, init=upscaled) → VAE decode. Изображение
/// condition'ит ОБЕ стадии (как офиц. combined_image_conditionings на каждой
/// resolution). `hp/wp` — сетка stage1 (выход = grid×64). `image` `[1,3,1,*,*]` [−1,1].
#[allow(clippy::too_many_arguments)]
pub fn generate_image_to_video_two_stage(
    dit: &VideoDit,
    encoder: &crate::vae::VaeEncoder,
    upscaler: &Upsampler,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    image: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    strength: f32,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    // image VAE-латент ×32 даёт сетку = (h_px/32, w_px/32). Для stage1 (hp×wp) и
    // stage2 (hp2×wp2) изображение энкодится в нужную сетку через resize латента?
    // нет — энкодер фиксирован ×32 от пикселей. Кодируем 1 раз, затем латент-сетка
    // совпадает с grid если image-пиксель = grid·32. Передаём image на stage2-разрешении
    // (hp·2·32) и для stage1 берём ту же сетку через pixel-resize вне функции — здесь
    // image уже на stage1-пикселях (hp·32×wp·32). stage2 кодирует upscale латента.
    let img_l1 = encoder.encode(image)?; // [1,128,1,hp,wp]
    let tok1 = frame_latent_to_tokens(&img_l1)?;
    let conds1 = vec![(0usize, tok1, strength)];
    let l1 = denoise_video_conditioned(dit, &ctx, fp, hp, wp, &DISTILLED_SIGMAS, &conds1, None, fps, device)?;
    // upscaler ×2 → stage2 (сетка hp2×wp2). image-латент для stage2 = upscale img_l1
    let l2 = upscaler.upsample(&l1)?.to_dtype(DType::BF16)?;
    let (hp2, wp2) = (hp * 2, wp * 2);
    let img_l2 = upscaler.upsample(&img_l1.to_dtype(DType::F32)?)?.to_dtype(DType::BF16)?; // [1,128,1,hp2,wp2]
    let tok2 = frame_latent_to_tokens(&img_l2)?;
    let conds2 = vec![(0usize, tok2, strength)];
    let latent = denoise_video_conditioned(dit, &ctx, fp, hp2, wp2, &STAGE2_SIGMAS, &conds2, Some(&l2), fps, device)?;
    vae.decode(&latent)
}

/// TI2Vid two-stage с guidance (полная не-distilled модель): guided stage1
/// (CFG+STG, `LTX2Scheduler` `num_steps`) → spatial-upscaler ×2 → distilled
/// refine stage2 (без guidance, `STAGE2_SIGMAS`) → VAE decode. Как официальный
/// `ti2vid_two_stages`: `dit1` (не-distilled, БЕЗ LoRA) для guided stage1,
/// `dit2` (= не-distilled + distilled-LoRA мерджнут) для refine stage2. Если
/// distilled-LoRA нет — передайте тот же dit обоими (`dit2=dit1`). Видео-поток.
#[allow(clippy::too_many_arguments)]
pub fn generate_ti2v_two_stage_video(
    dit1: &VideoDit,
    dit2: &VideoDit,
    upscaler: &Upsampler,
    vae: &VaeDecoder,
    video_encoding: &Tensor,
    neg_encoding: &Tensor,
    gp: &GuiderParams,
    num_steps: usize,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let latent = generate_ti2v_two_stage_video_latent(
        dit1, dit2, upscaler, video_encoding, neg_encoding, gp, num_steps, fp, hp, wp, fps, device,
    )?;
    vae.decode(&latent)
}

/// Как [`generate_ti2v_two_stage_video`], но возвращает ЛАТЕНТ без decode (для
/// дропа DiT перед VAE, см. [`generate_two_stage_latent`]).
#[allow(clippy::too_many_arguments)]
pub fn generate_ti2v_two_stage_video_latent(
    dit1: &VideoDit,
    dit2: &VideoDit,
    upscaler: &Upsampler,
    video_encoding: &Tensor,
    neg_encoding: &Tensor,
    gp: &GuiderParams,
    num_steps: usize,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    device: Device,
) -> Result<Tensor, LtxError> {
    let ctx = video_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let neg = neg_encoding.to_device(device)?.to_dtype(DType::BF16)?;
    let s1 = ltx2_sigmas(num_steps, fp * hp * wp);
    let l1 = denoise_video_guided(dit1, &ctx, &neg, gp, fp, hp, wp, &s1, None, fps, device, None, &DenoiseHooks::none())?;
    let l2 = upscaler.upsample(&l1)?.to_dtype(DType::BF16)?;
    let (hp2, wp2) = (hp * 2, wp * 2);
    denoise(dit2, &ctx, fp, hp2, wp2, &STAGE2_SIGMAS, Some(&l2), fps, device, None, &DenoiseHooks::none())
}

/// Совместная генерация видео+аудио (distilled, без CFG). Денойзит ОБА потока
/// через [`AvDit`] (8 шагов), декодирует видео ([`VaeDecoder`]) и аудио
/// ([`AudioVaeDecoder`]→log-mel→[`VocoderWithBwe`]). `fps` — кадр/с (для длины
/// аудио). Возвращает `(rgb [1,3,F,H,W], wave [1,2,L48k])`.
#[allow(clippy::too_many_arguments)]
pub fn generate_av(
    dit: &AvDit,
    vae: &VaeDecoder,
    audio_vae: &AudioVaeDecoder,
    vocoder: &VocoderWithBwe,
    video_encoding: &Tensor,
    audio_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    fps: f64,
    device: Device,
) -> Result<(Tensor, Tensor), LtxError> {
    let (v_latent, a_tok) = denoise_av(
        dit, video_encoding, audio_encoding, fp, hp, wp, &DISTILLED_SIGMAS, None, None, fps, device, None,
        &[], None, &DenoiseHooks::none(),
    )?;
    let rgb = vae.decode(&v_latent)?;
    let wave = decode_audio_tokens(audio_vae, vocoder, &a_tok)?;
    Ok((rgb, wave))
}

/// Число аудио-латент-токенов для видео из `fp` латент-кадров: F_pixel/fps · 25
/// латент/с (audio VAE 8 каналов × 16 мел).
pub fn audio_token_count(fp: usize, fps: f64) -> usize {
    let f_pixel = 8 * (fp.saturating_sub(1)) + 1;
    (((f_pixel as f64 / fps) * 25.0).round() as usize).max(1)
}

/// Аудио-токены `[1,Fa,128]` → волна `[1,2,L48k]` (unpatchify → audio-VAE → вокодер).
pub fn decode_audio_tokens(
    audio_vae: &AudioVaeDecoder,
    vocoder: &VocoderWithBwe,
    a_tok: &Tensor,
) -> Result<Tensor, LtxError> {
    let prof = crate::runtime::ltx_prof();
    let sync = || { let _ = synaptix_core::device::cuda::synchronize(0); };
    let fa = a_tok.dims()[1];
    let a_latent = a_tok.reshape(vec![1, fa, 8, 16])?.transpose(1, 2)?.contiguous()?.to_dtype(DType::F32)?;
    let t0 = std::time::Instant::now();
    let mel = audio_vae.decode(&a_latent)?; // [1,2,T,64]
    if prof { sync(); eprintln!("[LTX_PROF] audio-vae decode: {:.2}s (mel {:?})", t0.elapsed().as_secs_f32(), mel.dims()); }
    let t1 = std::time::Instant::now();
    let wave = vocoder.forward(&mel)?; // [1,2,L48k]
    if prof { sync(); eprintln!("[LTX_PROF] vocoder: {:.2}s (wave {:?})", t1.elapsed().as_secs_f32(), wave.dims()); }
    Ok(wave)
}

/// Joint A/V денойз-петля ([`AvDit`]) с опц. init (re-noise при σ₀, для stage2):
/// `v_init` латент `[1,128,F',H',W']`, `a_init` токены `[1,Fa,128]`. Возвращает
/// `(видео-латент f32 [1,128,F',H',W'], аудио-токены [1,Fa,128])` — decode у
/// вызывающего (позволяет дропнуть DiT перед VAE).
#[allow(clippy::too_many_arguments)]
pub fn denoise_av(
    dit: &AvDit,
    video_encoding: &Tensor,
    audio_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    v_init: Option<&Tensor>,
    a_init: Option<&Tensor>,
    fps: f64,
    device: Device,
    v_nag: Option<(&Tensor, f32, f32, f32)>,
    v_conds: &[(usize, Tensor, f32)],
    seed: Option<u64>,
    hooks: &DenoiseHooks,
) -> Result<(Tensor, Tensor), LtxError> {
    let tv = fp * hp * wp;
    let v_pos = pixel_coords(fp, hp, wp, fps);
    let ddt = dit.compute_dtype();
    let v_ctx = video_encoding.to_device(device)?.to_dtype(ddt)?;
    let a_ctx = audio_encoding.to_device(device)?.to_dtype(ddt)?;
    let fa = audio_token_count(fp, fps);
    let a_pos = audio_coords(fa);
    let s0 = sigmas[0];

    // начальные латенты: шум, либо re-noise noise·σ₀ + init·(1−σ₀) (GaussianNoiser)
    let v_noise = noise_tensor(vec![1usize, 128, fp, hp, wp], seed)?.to_device(device)?.to_dtype(ddt)?;
    let a_noise = noise_tensor(vec![1usize, 8, fa, 16], seed)?.to_device(device)?.to_dtype(ddt)?;
    let mut v_tok = v_noise.reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?; // [1,Tv,128]
    if let Some(init) = v_init {
        let clean = init.to_device(device)?.to_dtype(ddt)?
            .reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?;
        v_tok = v_tok.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?;
    }
    let mut a_tok = a_noise.transpose(1, 2)?.contiguous()?.reshape(vec![1, fa, 128])?; // b c t f→b t (c f)
    if let Some(init) = a_init {
        let clean = init.to_device(device)?.to_dtype(ddt)?;
        a_tok = a_tok.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?;
    }

    // image/keyframe-conditioning видео-потока (replace-latent, bit-faithful к
    // denoise_video_conditioned): mask=1−strength на conditioned-позициях, замена
    // их токенов на image-латент, per-token timesteps + blend в петле. Аудио-поток
    // conditioning не несёт (i2v кондишенит только видео). conds пуст → text2video
    // путь нетронут (ветка `cond` ниже не активна).
    let cond = !v_conds.is_empty();
    let mut v_mask_v = vec![1f32; tv];
    let (mut v_mask, mut v_inv_mask, mut v_clean) = (None, None, None);
    if cond {
        for (start, toks, strength) in v_conds {
            let n = toks.dims()[1];
            for k in *start..(*start + n).min(tv) {
                v_mask_v[k] = 1.0 - *strength;
            }
        }
        for (start, toks, _) in v_conds {
            let toks = toks.to_device(device)?.to_dtype(ddt)?;
            let n = toks.dims()[1];
            let pre = if *start > 0 { Some(v_tok.narrow(1, 0, *start)?.contiguous()?) } else { None };
            let post = if start + n < tv {
                Some(v_tok.narrow(1, start + n, tv - start - n)?.contiguous()?)
            } else {
                None
            };
            let mut parts: Vec<Tensor> = Vec::new();
            if let Some(p) = &pre { parts.push(p.clone()); }
            parts.push(toks.clone());
            if let Some(p) = &post { parts.push(p.clone()); }
            let refs: Vec<&Tensor> = parts.iter().collect();
            v_tok = Tensor::cat(&refs, 1)?.contiguous()?;
        }
        let m = Tensor::from_vec(v_mask_v.clone(), vec![1, tv, 1], device)?.to_dtype(ddt)?;
        v_inv_mask = Some(m.affine(-1.0, 1.0)?);
        v_clean = Some(v_tok.clone());
        v_mask = Some(m);
    }

    let prof = crate::runtime::ltx_prof();
    let ord = if let Device::Cuda(o) = device { o } else { 0 };
    let sync = || { let _ = synaptix_core::device::cuda::synchronize(ord); };
    let max_steps = sigmas.len() - 1;
    // Компакт пула на границе шага: живых тензоров минимум (латенты+ctx) →
    // trim возвращает драйверу почти все сегменты. Иначе фрагментация копится
    // (на 20s-refine live 14GB размазывался по 24.8GB reserved-сегментов —
    // trim внутри шага не возвращал НИ ОДНОГО, OOM при free=0.04GB).
    let compact = || {
        if matches!(device, Device::Cuda(_)) && tv > 32768 {
            let _ = synaptix_core::device::cuda::synchronize_all(ord);
            let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(ord);
        }
    };
    // На время длинного denoise пул ДЕРЖИТ свободные блоки (trim-threshold
    // высокий): агрессивный возврат драйверу разрушал free-list — пул строил
    // сегменты заново и interleaving живых [T,dim] с транзиентами давал
    // устойчивую фрагментацию (nvfp4-19s: 10.8GB свободно внутри пула, куска
    // 418MB нет, 5 ретраев бессильны). Реюз стабильных size-классов лечит.
    // Перед выходом threshold сбрасывается и всё возвращается разом (VAE).
    let long_t = matches!(device, Device::Cuda(_)) && tv > 32768;
    if long_t {
        synaptix_core::memory::cuda_pool::set_trim_threshold(22 << 30);
    }
    for i in 0..max_steps {
        if hooks.cancelled() {
            if long_t {
                synaptix_core::memory::cuda_pool::set_trim_threshold(0);
            }
            compact();
            return Err(LtxError::Cancelled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        // per-token timesteps видео: при conditioning ts = mask·sigma (conditioned
        // позиции «уже чистые» → меньший timestep), иначе равномерно sigma.
        let v_ts: Vec<f32> = if cond {
            v_mask_v.iter().map(|&m| m * sigma as f32).collect()
        } else {
            vec![sigma as f32; tv]
        };
        let a_ts = vec![sigma as f32; fa];
        let t_step = std::time::Instant::now();
        // компакт только перед ПЕРВЫМ шагом (после upsample-каши); на остальных
        // границах weights-пул держит карусель, а sync+trim стоит ~1s/шаг
        if i == 0 { compact(); }
        let (v_vel, a_vel) = dit.forward(
            &v_tok, &v_ts, sigma as f32, &v_pos, &v_ctx,
            &a_tok, &a_ts, sigma as f32, &a_pos, &a_ctx,
            v_nag,
        )?;
        let dt = (sigma_next - sigma) as f32;
        if cond {
            // per-token euler видео + blend conditioned-позиций к clean (frozen).
            let m = v_mask.as_ref().unwrap();
            let inv = v_inv_mask.as_ref().unwrap();
            let clean = v_clean.as_ref().unwrap();
            let vdt = m.mul_scalar(dt)?; // [1,tv,1]
            v_tok = v_tok.add(&v_vel.broadcast_mul(&vdt)?)?;
            v_tok = v_tok.broadcast_mul(m)?.add(&clean.broadcast_mul(inv)?)?;
        } else {
            v_tok = v_tok.add(&v_vel.mul_scalar(dt)?)?;
        }
        a_tok = a_tok.add(&a_vel.mul_scalar(dt)?)?;
        if prof { sync(); eprintln!("[LTX_PROF] av step {i} (Tv {tv} Ta {fa}): {:.2}s", t_step.elapsed().as_secs_f32()); }
        hooks.emit(i + 1, max_steps, sigma_next);
    }
    // финальный компакт: threshold сбрасывается, транзиенты последнего шага
    // освобождены → пул возвращается драйверу, даунстрим (VAE) видит честный free
    if long_t {
        synaptix_core::memory::cuda_pool::set_trim_threshold(0);
    }
    compact();
    let v_latent = v_tok.transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?.to_dtype(DType::F32)?;
    Ok((v_latent, a_tok))
}

/// RGB `[1,3,F,H,W]` (≈[−1,1]) → кадры `[3,H,W]` в [0,1] для записи видео.
pub fn rgb_to_frames(rgb: &Tensor) -> Result<Vec<Tensor>, LtxError> {
    let (f, h, w) = (rgb.dims()[2], rgb.dims()[3], rgb.dims()[4]);
    // (x+1)/2, clamp[0,1]
    let scaled = rgb.affine(0.5, 0.5)?.to_dtype(DType::F32)?;
    let mut frames = Vec::with_capacity(f);
    for fi in 0..f {
        let fr = scaled
            .narrow(2, fi, 1)? // [1,3,1,H,W]
            .contiguous()?
            .reshape(vec![3, h, w])?
            .clamp(0.0, 1.0)?
            .contiguous()?;
        frames.push(fr);
    }
    Ok(frames)
}

#[cfg(test)]
mod sched_tests {
    use super::ltx2_sigmas;

    fn close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() < 1e-4, "{x} vs {y}");
        }
    }

    #[test]
    fn ltx2_matches_python() {
        // эталон LTX2Scheduler.execute (tokens=4096 = default_number_of_tokens)
        close(&ltx2_sigmas(8, 4096), &[1.0, 0.96571, 0.92187, 0.86386, 0.78345, 0.66458, 0.471, 0.1, 0.0]);
        close(&ltx2_sigmas(4, 4096), &[1.0, 0.86708, 0.63157, 0.1, 0.0]);
        // монотонно убывает, начинается 1.0, заканчивается 0.0
        let s = ltx2_sigmas(40, 1024);
        assert_eq!(s[0], 1.0);
        assert_eq!(*s.last().unwrap(), 0.0);
        assert!(s.windows(2).all(|w| w[0] >= w[1]));
    }
}

/// Позиции audio-reference для lipdub: [`audio_coords`]`(fa)`, сдвинутые в
/// ОТРИЦАТЕЛЬНУЮ зону (`pos − aud_dur − 0.04`, aud_dur = конец последнего токена)
/// — reference-токены живут «до» генерируемого аудио (bit-faithful к
/// `patchify_lipdub_audio_reference_latent(negative_positions=True)`).
pub fn lipdub_audio_ref_positions(fa: usize) -> Vec<f64> {
    let mut p = audio_coords(fa);
    let aud_dur = p[(fa - 1) * 2 + 1]; // max end
    for v in p.iter_mut() {
        *v -= aud_dur + 0.04;
    }
    p
}

/// Joint A/V денойз с APPEND-conditioning на обоих потоках (lipdub):
/// видео = `[main (Tv) ++ v_app]` (ref-видео, mask 0), аудио = `[main (Fa) ++ a_app]`
/// (ref-аудио, mask 0). `v_init`/`a_init` — re-noise main-части при σ₀ (stage2).
/// `a_frozen` — main-аудио заморожено (mask 0, stage2: контекст для видео).
/// Возвращает (видео-латент f32, main-аудио токены).
#[allow(clippy::too_many_arguments)]
pub fn denoise_av_append(
    dit: &AvDit,
    video_encoding: &Tensor,
    audio_encoding: &Tensor,
    fp: usize,
    hp: usize,
    wp: usize,
    sigmas: &[f64],
    v_init: Option<&Tensor>,
    a_init: Option<&Tensor>,
    a_frozen: bool,
    v_app: Option<(&Tensor, &[f64], f32)>,
    a_app: Option<(&Tensor, &[f64])>,
    fps: f64,
    device: Device,
    seed: Option<u64>,
    hooks: &DenoiseHooks,
) -> Result<(Tensor, Tensor), LtxError> {
    let tv = fp * hp * wp;
    let ddt = dit.compute_dtype();
    let v_ctx = video_encoding.to_device(device)?.to_dtype(ddt)?;
    let a_ctx = audio_encoding.to_device(device)?.to_dtype(ddt)?;
    let fa = audio_token_count(fp, fps);
    let s0 = sigmas[0];

    // main-части (шум / re-noise)
    let v_noise = noise_tensor(vec![1usize, 128, fp, hp, wp], seed)?.to_device(device)?.to_dtype(DType::BF16)?;
    let mut v_main = v_noise.reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?;
    if let Some(init) = v_init {
        let clean = init.to_device(device)?.to_dtype(ddt)?
            .reshape(vec![1, 128, tv])?.transpose(1, 2)?.contiguous()?;
        v_main = v_main.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?;
    }
    let a_noise = noise_tensor(vec![1usize, 8, fa, 16], seed)?.to_device(device)?.to_dtype(DType::BF16)?;
    let mut a_main = a_noise.transpose(1, 2)?.contiguous()?.reshape(vec![1, fa, 128])?;
    if let Some(init) = a_init {
        let clean = init.to_device(device)?.to_dtype(DType::BF16)?;
        a_main = if a_frozen {
            clean // frozen: чистый init без шума (noise_scale=0)
        } else {
            a_main.mul_scalar(s0 as f32)?.add(&clean.mul_scalar((1.0 - s0) as f32)?)?
        };
    }

    // сборка последовательностей: main ++ append (append fixed, mask 0)
    let (mut v_tok, v_pos, tv_all, v_mask) = match &v_app {
        Some((app, app_pos, strength)) => {
            let app = app.to_device(device)?.to_dtype(DType::BF16)?;
            let ta = app.dims()[1];
            let pos = cat_positions(&pixel_coords(fp, hp, wp, fps), tv, app_pos, ta);
            let mut m = vec![1f32; tv];
            m.extend(std::iter::repeat(1.0 - *strength).take(ta));
            (Tensor::cat(&[&v_main, &app], 1)?.contiguous()?, pos, tv + ta, m)
        }
        None => (v_main, pixel_coords(fp, hp, wp, fps), tv, vec![1f32; tv]),
    };
    let a_main_mask = if a_frozen { 0f32 } else { 1f32 };
    let (mut a_tok, a_pos, fa_all, a_mask) = match &a_app {
        Some((app, app_pos)) => {
            let app = app.to_device(device)?.to_dtype(DType::BF16)?;
            let ta = app.dims()[1];
            let mut pos = audio_coords(fa);
            pos.extend_from_slice(app_pos);
            let mut m = vec![a_main_mask; fa];
            m.extend(std::iter::repeat(0f32).take(ta));
            (Tensor::cat(&[&a_main, &app], 1)?.contiguous()?, pos, fa + ta, m)
        }
        None => (a_main, audio_coords(fa), fa, vec![a_main_mask; fa]),
    };
    let vm = Tensor::from_vec(v_mask.clone(), vec![1, tv_all, 1], device)?.to_dtype(DType::BF16)?;
    let vm_inv = vm.affine(-1.0, 1.0)?;
    let am = Tensor::from_vec(a_mask.clone(), vec![1, fa_all, 1], device)?.to_dtype(DType::BF16)?;
    let am_inv = am.affine(-1.0, 1.0)?;
    let v_clean = v_tok.clone();
    let a_clean = a_tok.clone();

    for i in 0..sigmas.len() - 1 {
        if hooks.cancelled() {
            return Err(LtxError::Cancelled);
        }
        let (sigma, sigma_next) = (sigmas[i], sigmas[i + 1]);
        let v_ts: Vec<f32> = v_mask.iter().map(|&m| m * sigma as f32).collect();
        let a_ts: Vec<f32> = a_mask.iter().map(|&m| m * sigma as f32).collect();
        let (v_vel, a_vel) = dit.forward(
            &v_tok, &v_ts, sigma as f32, &v_pos, &v_ctx,
            &a_tok, &a_ts, sigma as f32, &a_pos, &a_ctx,
            None,
        )?;
        let dt = (sigma_next - sigma) as f32;
        v_tok = v_tok.add(&v_vel.broadcast_mul(&vm.mul_scalar(dt)?)?)?;
        v_tok = v_tok.broadcast_mul(&vm)?.add(&v_clean.broadcast_mul(&vm_inv)?)?;
        a_tok = a_tok.add(&a_vel.broadcast_mul(&am.mul_scalar(dt)?)?)?;
        a_tok = a_tok.broadcast_mul(&am)?.add(&a_clean.broadcast_mul(&am_inv)?)?;
        hooks.emit(i + 1, sigmas.len() - 1, sigma_next);
    }
    let v_latent = v_tok.narrow(1, 0, tv)?.contiguous()?
        .transpose(1, 2)?.contiguous()?.reshape(vec![1, 128, fp, hp, wp])?.to_dtype(DType::F32)?;
    let a_out = a_tok.narrow(1, 0, fa)?.contiguous()?;
    Ok((v_latent, a_out))
}

/// LipDub (официальный поток): stage1 joint A/V denoise с ref-видео (append,
/// маска 0) + ref-аудио (append, отриц. позиции) → upscale видео ×2 → stage2
/// refine видео (ref-видео на полном разрешении; аудио ЗАМОРОЖЕНО = результат
/// stage1, контекст для видео). LipDub IC-LoRA должна быть смерджена в `dit`.
/// `refN`/`refN_pos` — токены+позиции reference-видео на разрешении стадии N.
/// Возвращает (видео-латент full-res f32, аудио-токены stage1).
#[allow(clippy::too_many_arguments)]
pub fn generate_lipdub_latents(
    dit: &AvDit,
    upscaler: &Upsampler,
    video_encoding: &Tensor,
    audio_encoding: &Tensor,
    ref1: &Tensor,
    ref1_pos: &[f64],
    ref2: &Tensor,
    ref2_pos: &[f64],
    audio_ref: &Tensor,
    fp: usize,
    hp1: usize,
    wp1: usize,
    fps: f64,
    device: Device,
) -> Result<(Tensor, Tensor), LtxError> {
    let far = audio_ref.dims()[1];
    let a_ref_pos = lipdub_audio_ref_positions(far);
    // stage1: joint, оба reference
    let (l1, a1) = denoise_av_append(
        dit, video_encoding, audio_encoding, fp, hp1, wp1, &DISTILLED_SIGMAS,
        None, None, false,
        Some((ref1, ref1_pos, 1.0)), Some((audio_ref, &a_ref_pos)),
        fps, device, None, &DenoiseHooks::none(),
    )?;
    // upscale видео ×2 → stage2: видео refine, аудио frozen=stage1 + его же ref-копия
    let l2 = upscaler.upsample(&l1)?.to_dtype(DType::BF16)?;
    let fa1 = a1.dims()[1];
    let a1_ref_pos = lipdub_audio_ref_positions(fa1);
    let (latent, _a2) = denoise_av_append(
        dit, video_encoding, audio_encoding, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS,
        Some(&l2), Some(&a1), true,
        Some((ref2, ref2_pos, 1.0)), Some((&a1, &a1_ref_pos)),
        fps, device, None, &DenoiseHooks::none(),
    )?;
    Ok((latent, a1))
}

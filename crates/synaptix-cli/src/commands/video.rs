//! `synaptix video` — генерация видео (+аудио) LTX-2.3 по текстовому промпту.
//!
//! Пайплайн: живой Gemma-3-12B (text→49 hidden states) → Video/Audio
//! text-conditioner → DiT (AvDit/VideoDit, streaming-offload) → VAE-декод
//! (пространственный авто-тайлинг, FullHD без OOM) → [audio VAE + вокодер] →
//! ffmpeg-мукс в mp4. Per-component квант как в LLM: `--quant-transformer`
//! (блоки DiT), `--quant-encoder` (Gemma), `--compute-dtype`.

use std::path::PathBuf;
use std::process::Command;

use synaptix_core::precision::PrecisionConfig;
use synaptix_core::{device::Device, dtype::DType};
use synaptix_llm_gemma3::pipeline::GemmaPipeline;
use synaptix_video_ltx23::dit::{AvDit, VideoDit};
use synaptix_video_ltx23::loader::{LoraWeights, LtxCheckpoint};
use synaptix_video_ltx23::guider::GuiderParams;
use synaptix_video_ltx23::spec;
use synaptix_video_ltx23::audio_vae::{ltx_log_mel, AudioVaeEncoder};
use synaptix_core::tensor::Tensor;
use synaptix_video_ltx23::pipeline::{
    audio_token_count, decode_audio_tokens, denoise, denoise_av, denoise_av_append, denoise_av_retake,
    fp_for_frames, frame_latent_to_tokens, frames_for_duration, keyframe_positions, latent_grid,
    lipdub_audio_ref_positions, out_frame_count, ref_video_positions, rgb_to_frames, stage1_grid,
    DenoiseHooks, DenoiseProgress, DISTILLED_SIGMAS, STAGE2_SIGMAS, SUPPORTED_FPS,
};
use synaptix_video_ltx23::text_encoder::{AudioTextConditioner, VideoTextConditioner};
use synaptix_video_ltx23::upscaler::Upsampler;
use synaptix_video_ltx23::vae::{VaeDecoder, VaeEncoder};
use synaptix_video_ltx23::audio_vae::AudioVaeDecoder;
use synaptix_video_ltx23::vocoder::VocoderWithBwe;

pub struct VideoArgs {
    pub model: PathBuf,   // LTX-2.3 .safetensors (DiT+VAE+vocoder+проекции)
    pub prompt: String,   // сцена + (для аудио) описание звука/речи
    pub output: PathBuf,
    pub gemma: PathBuf,   // директория Gemma-3-12B
    pub frames: Option<usize>, // явные кадры; переопределяет duration
    pub duration: String,      // «10s» | «2.5s» | «1m» | секунды числом
    pub width: usize,
    pub height: usize,
    pub fps: f64,
    pub no_audio: bool,
    pub pipeline: Option<String>,  // имя пресета (--pipeline); переопределяет two_stage/no_audio
    pub list_pipelines: bool,      // напечатать реестр и выйти
    pub two_stage: bool,            // stage1 → upscaler ×2 → stage2-refine (видео)
    pub upscaler: Option<PathBuf>, // spatial-upscaler ×2 .safetensors
    pub no_refine: bool,           // two-stage без stage2-refine
    pub lora: Option<PathBuf>,     // distilled-LoRA для мерджа в DiT
    pub lora_strength: f32,
    pub lora_strength_stage1: Option<f32>,
    pub lora_strength_stage2: Option<f32>, // two-stage: на stage2 (HQ 0.5/ti2v 0.8); деф. lora_strength
    pub negative_prompt: String,   // CFG negative (guidance)
    pub cfg_scale: f32,            // CFG scale (1.0 = выкл)
    pub stg_scale: f32,           // STG scale (0.0 = выкл)
    pub steps: usize,             // шаги guided stage1
    pub image: Option<PathBuf>,    // conditioning-кадр (image→video)
    pub image_strength: f32,
    pub image_frame: usize,        // 0=replace, >0=keyframe append
    pub video: Option<PathBuf>,    // исходное видео для retake
    pub retake_start: f64,
    pub retake_end: f64,
    pub ref_video: Option<PathBuf>, // IC-LoRA reference (control)
    pub ref_downscale: usize,
    pub ref_strength: f32,
    pub audio: Option<PathBuf>,    // речь для lipdub (с --ref-video)
    pub ref_preprocess: String,    // none | canny | depth (control-сигнал из ref-видео)
    pub canny_low: f32,
    pub canny_high: f32,
    pub depth_model: PathBuf,      // Depth Anything V2 (для depth)
    pub quant_transformer: Option<String>,
    pub quant_encoder: Option<String>,
    pub compute_dtype: Option<String>,
    pub device: String,
    pub nag_prompt: Option<String>, // NAG negative-prompt (вкл. NAG на stage1 v_attn2)
    pub nag_scale: f32,
    pub nag_alpha: f32,
    pub nag_tau: f32,
    pub force_offload: bool, // принудительный host-stream offload квантованного DiT
    pub prof: bool,          // печать таймингов text-encoding ([LTX_PROF])
    pub block_mode: Option<usize>, // dense-offload: 0=легаси-карусель, 1=слоты, 2=слоты+graph
    pub seed: Option<u64>,
    pub cfg_rescale: f32,
    pub stg_blocks: Vec<usize>,
    pub guider_skip_step: usize,
    pub control_preview: Option<PathBuf>,
    pub crf: Option<u32>,
}

enum Mode {
    Text,
    Image { path: PathBuf, strength: f32, frame: usize },
    Retake { path: PathBuf, start: f64, end: f64 },
    IcLora { path: PathBuf },
    A2v { audio: PathBuf },
    Lipdub { reference: PathBuf, audio: PathBuf },
}

impl Mode {
    fn from_args(args: &VideoArgs) -> Result<Self, String> {
        let exclusive = [args.image.is_some(), args.video.is_some(), args.ref_video.is_some()]
            .iter()
            .filter(|b| **b)
            .count();
        if exclusive > 1 {
            return Err("--image, --video и --ref-video взаимоисключающие".into());
        }
        Ok(match (&args.ref_video, &args.audio, &args.video, &args.image) {
            (Some(r), Some(a), _, _) => Mode::Lipdub { reference: r.clone(), audio: a.clone() },
            (Some(r), None, _, _) => Mode::IcLora { path: r.clone() },
            (None, Some(a), None, None) => Mode::A2v { audio: a.clone() },
            (None, Some(_), _, _) => return Err("--audio без --ref-video работает только как audio→video (без --video/--image)".into()),
            (None, None, Some(v), _) => Mode::Retake { path: v.clone(), start: args.retake_start, end: args.retake_end },
            (None, None, None, Some(i)) => {
                Mode::Image { path: i.clone(), strength: args.image_strength, frame: args.image_frame }
            }
            (None, None, None, None) => Mode::Text,
        })
    }

    fn conditioning(&self) -> spec::Conditioning {
        match self {
            Mode::Text => spec::Conditioning::None,
            Mode::Image { frame: 0, .. } => spec::Conditioning::Image,
            Mode::Image { .. } => spec::Conditioning::Keyframe,
            Mode::Retake { .. } => spec::Conditioning::RetakeMask,
            Mode::IcLora { .. } | Mode::Lipdub { .. } => spec::Conditioning::Video,
            Mode::A2v { .. } => spec::Conditioning::AudioInput,
        }
    }

    fn adapter_lora(&self) -> bool {
        matches!(self, Mode::IcLora { .. } | Mode::Lipdub { .. })
    }

    fn label(&self) -> &'static str {
        match self {
            Mode::Text => "text→video",
            Mode::Image { frame: 0, .. } => "image→video",
            Mode::Image { .. } => "keyframe",
            Mode::Retake { .. } => "retake",
            Mode::IcLora { .. } => "ic-lora",
            Mode::A2v { .. } => "audio→video",
            Mode::Lipdub { .. } => "lipdub",
        }
    }
}

type R<T> = Result<T, Box<dyn std::error::Error>>;

fn progress_printer(label: &'static str) -> impl Fn(DenoiseProgress) + Sync {
    move |p: DenoiseProgress| {
        eprint!("\r  {label}: шаг {}/{} (σ {:.3})    ", p.step + 1, p.total, p.sigma);
        if p.step + 1 == p.total {
            eprintln!();
        }
    }
}

fn load_upsampler(ckpt: &LtxCheckpoint, path: &std::path::Path, dev: Device) -> R<Upsampler> {
    let view = ckpt.view_on(dev);
    let mean = view.get_raw("vae.per_channel_statistics.mean-of-means").map_err(|e| format!("vae mean: {e}"))?;
    let std = view.get_raw("vae.per_channel_statistics.std-of-means").map_err(|e| format!("vae std: {e}"))?;
    Ok(Upsampler::load(path, &mean, &std, dev).map_err(|e| format!("upscaler: {e}"))?)
}

fn image_tokens(encoder: &VaeEncoder, image: &Tensor, hp: usize, wp: usize, dev: Device) -> R<Tensor> {
    let (ph, pw) = (hp * 32, wp * 32);
    let img = synaptix_io::image::fit_image(image, pw, ph, false)?
        .to_device(dev)?
        .affine(2.0, -1.0)?
        .contiguous()?
        .reshape(vec![1, 3, 1, ph, pw])?;
    let latent = encoder.encode(&img).map_err(|e| format!("VAE encode: {e}"))?;
    Ok(frame_latent_to_tokens(&latent)?)
}

fn latent_tokens(latent: &Tensor) -> R<(Tensor, usize, usize, usize)> {
    let (fpr, hr, wr) = (latent.dims()[2], latent.dims()[3], latent.dims()[4]);
    let tok = latent.reshape(vec![1, 128, fpr * hr * wr])?.transpose(1, 2)?.contiguous()?;
    Ok((tok, fpr, hr, wr))
}

fn audio_input_latent(ckpt: &LtxCheckpoint, path: &std::path::Path, fa: usize, dev: Device) -> R<Tensor> {
    let aenc = AudioVaeEncoder::load(ckpt, dev).map_err(|e| format!("audio encoder: {e}"))?;
    let mel = ltx_log_mel(&[load_audio_16k(path)?], dev).map_err(|e| format!("mel: {e}"))?;
    let enc = aenc.encode(&mel).map_err(|e| format!("audio encode: {e}"))?;
    let far = enc.dims()[1];
    if far >= fa {
        return Ok(enc.narrow(1, 0, fa)?.contiguous()?);
    }
    let last = enc.narrow(1, far - 1, 1)?;
    let mut parts = vec![enc];
    parts.extend((far..fa).map(|_| last.clone()));
    let refs: Vec<&Tensor> = parts.iter().collect();
    Ok(Tensor::cat(&refs, 1)?.contiguous()?)
}

fn release_vram(dev: Device) {
    if let Device::Cuda(o) = dev {
        let _ = synaptix_core::device::cuda::synchronize_all(o);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(o);
    }
}


fn parse_duration_secs(s: &str) -> Result<f64, String> {
    let t = s.trim();
    let (num, mult) = if let Some(x) = t.strip_suffix('s') {
        (x, 1.0)
    } else if let Some(x) = t.strip_suffix('m') {
        (x, 60.0)
    } else {
        (t, 1.0)
    };
    let v: f64 = num.trim().parse().map_err(|_| {
        format!("неразборчивая --duration «{s}» (примеры: 10s, 2.5s, 1m, 7)")
    })?;
    if !v.is_finite() || v <= 0.0 {
        return Err(format!("--duration должна быть > 0, получено «{s}»"));
    }
    Ok(v * mult)
}

fn parse_quant(s: Option<&str>, default: DType, dense: DType) -> Result<DType, String> {
    match s.map(|x| x.to_lowercase()) {
        None => Ok(default),
        Some(q) => match q.as_str() {
            "none" | "dense" => Ok(dense),
            "bf16" => Ok(DType::BF16),
            "f16" => Ok(DType::F16),
            "nvfp4" => Ok(DType::NVFP4),
            "mxfp8" | "fp8" => Ok(DType::MXFP8),
            other => Err(format!("неизвестный квант: {other} (none|mxfp8|nvfp4)")),
        },
    }
}

pub fn run(args: VideoArgs) -> R<()> {
    if args.prof {
        synaptix_video_ltx23::runtime::set_ltx_prof(true);
    }
    if let Some(m) = args.block_mode {
        synaptix_video_ltx23::runtime::set_ltx_block_mode(m);
    }
    if args.list_pipelines {
        println!("Пайплайны LTX-2.3 (--pipeline <name>):");
        for p in spec::registry() {
            let st = if p.implemented() { "✓".to_string() } else { format!("Фаза {}", p.todo_phase.unwrap()) };
            println!("  {:<16} [{st:<7}] {}", p.name, p.desc);
        }
        return Ok(());
    }

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let dev = crate::commands::device::resolve(&args.device);
    let ord = if let Device::Cuda(o) = dev { o } else { 0 };
    let compute = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") | None => DType::BF16,
        Some("f32") => DType::F32,
        Some(o) => return Err(format!("unknown compute-dtype {o} (f16|bf16|f32)").into()),
    };
    let quant_dit = parse_quant(args.quant_transformer.as_deref(), compute, compute)?;
    let quant_enc = parse_quant(args.quant_encoder.as_deref(), DType::MXFP8, compute)?;
    let seed = args.seed.filter(|&s| s != 0);

    if !SUPPORTED_FPS.contains(&args.fps) {
        return Err(format!("--fps {} не поддерживается (24 | 25 | 48 | 50)", args.fps).into());
    }
    let mode = Mode::from_args(&args)?;
    let (hp, wp) = latent_grid(args.width, args.height);
    let fp = match args.frames {
        Some(f) => fp_for_frames(f),
        None => frames_for_duration(parse_duration_secs(&args.duration)?, args.fps),
    };
    let out_frames = out_frame_count(fp);
    let sp = match &args.pipeline {
        Some(name) => {
            let sp = spec::by_name(name).ok_or_else(|| {
                format!("неизвестный --pipeline {name} (доступно: {})", spec::names().join(", "))
            })?;
            let want = mode.conditioning();
            let ok = sp.conditioning == want
                || (sp.conditioning == spec::Conditioning::Image && want == spec::Conditioning::Keyframe)
                || (sp.conditioning == spec::Conditioning::Keyframe && want == spec::Conditioning::Image);
            if !ok {
                return Err(format!(
                    "--pipeline {} ждёт условие {:?}, а по флагам выходит {} ({want:?})",
                    sp.name,
                    sp.conditioning,
                    mode.label()
                )
                .into());
            }
            sp
        }
        None => {
            let derived = match &mode {
                Mode::Lipdub { .. } => "lipdub",
                Mode::Retake { .. } => "retake",
                Mode::A2v { .. } if args.two_stage || args.upscaler.is_some() => "a2v",
                _ if args.two_stage => "two-stage",
                Mode::Text if args.no_audio => "one-stage",
                _ => "av",
            };
            spec::by_name(derived).expect("derived pipeline known")
        }
    };
    let mut two_stage = sp.stages == spec::Stages::Two;
    if matches!(mode, Mode::Retake { .. }) && two_stage {
        eprintln!("  [внимание] retake идёт одной стадией на целевом разрешении");
        two_stage = false;
    }
    let refine = sp.refine && !args.no_refine;
    let guided = args.cfg_scale > 1.0 || args.stg_scale > 0.0;
    if guided && !two_stage {
        return Err("guidance (--cfg-scale/--stg-scale) поддержан только на two-stage (guided stage1)".into());
    }
    if guided && !matches!(mode, Mode::Text) {
        return Err("guidance (--cfg-scale/--stg-scale) поддержан только для text→video".into());
    }
    let keep_audio = sp.modality == spec::Modality::AudioVideo && !args.no_audio && !guided;
    let av_path = !guided && (keep_audio || !matches!(mode, Mode::Text));
    let need_actx = av_path;
    if guided && !args.no_audio {
        eprintln!("  [внимание] guided-путь только видео → аудио отключено");
    }
    if two_stage && args.upscaler.is_none() {
        return Err(format!("{} на двух стадиях требует --upscaler <spatial-upscaler>", mode.label()).into());
    }
    if mode.adapter_lora() && args.lora.is_none() {
        eprintln!("  [внимание] {} рассчитан на IC-LoRA — укажи --lora", mode.label());
    }
    let guider = GuiderParams {
        cfg_scale: args.cfg_scale,
        stg_scale: args.stg_scale,
        rescale_scale: if guided { args.cfg_rescale } else { 0.0 },
        modality_scale: 1.0,
        skip_step: args.guider_skip_step as u32,
        stg_blocks: args.stg_blocks.clone(),
    };
    let ckpt = LtxCheckpoint::open(&args.model, Device::Cpu, DType::BF16).map_err(|e| format!("LTX ckpt: {e}"))?;
    let offload = if quant_dit == compute {
        true
    } else if args.force_offload {
        true
    } else if let Device::Cuda(o) = dev {
        let dit_b = synaptix_video_ltx23::dit::dit_resident_bytes(&ckpt, quant_dit, compute);
        let vae_b: usize = ckpt
            .infos()
            .filter(|(n, _, _)| n.starts_with("vae."))
            .map(|(_, _, s)| compute.bytes_for_numel(s.iter().product()))
            .sum();
        let act_b = fp * hp * wp * 4096 * 2 * 28;
        let quant_pad_b = (fp * hp * wp + 256) * (16384 * 2 + 16384 * 2 + 12288 * 2);
        let need = dit_b + vae_b + act_b + quant_pad_b + (1usize << 30);
        let (free, _total) = synaptix_core::device::cuda::mem_info(o).map_err(|e| format!("mem_info: {e}"))?;
        if need > free {
            eprintln!(
                "  DiT {quant_dit:?} резидентно ~{:.1}GB (+VAE/активации) > свободно {:.1}GB → streaming-offload из host-RAM",
                dit_b as f64 / 1e9,
                free as f64 / 1e9,
            );
            true
        } else {
            false
        }
    } else {
        false
    };
    eprintln!(
        "synaptix video [{} / {}]: «{}»\n  {}×{} (сетка {hp}×{wp}) | {out_frames} кадров @ {}fps (~{:.1}s) | аудио={keep_audio} | two_stage={two_stage} refine={refine} | guided={guided}{} | seed {}\n  DiT quant={quant_dit:?} (offload={offload}) | Gemma quant={quant_enc:?} | compute={compute:?} | {dev:?}",
        sp.name,
        mode.label(),
        args.prompt,
        wp * 32,
        hp * 32,
        args.fps,
        out_frames as f64 / args.fps,
        if guided { format!(" (cfg {} stg {} steps {})", args.cfg_scale, args.stg_scale, args.steps) } else { String::new() },
        seed.map(|s| s.to_string()).unwrap_or_else(|| "случайный".into()),
    );
    let _pin_cache = if offload && quant_dit == compute && matches!(dev, Device::Cuda(_)) {
        Some(if quant_enc.is_quantized() {
            synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused(&ckpt.shard_bytes())
        } else {
            synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused_lazy(&ckpt.shard_bytes())
        })
    } else {
        None
    };

    let lora_single = args.lora_strength;
    let (lora_s1, lora_s2) = if mode.adapter_lora() {
        (
            args.lora_strength_stage1.unwrap_or(args.lora_strength),
            args.lora_strength_stage2.unwrap_or(args.lora_strength),
        )
    } else {
        (
            args.lora_strength_stage1.unwrap_or(0.0),
            args.lora_strength_stage2.unwrap_or(args.lora_strength),
        )
    };
    if let Some(lp) = &args.lora {
        if two_stage {
            eprintln!("  LoRA: {} (stage1 {lora_s1} / stage2 {lora_s2})", lp.display());
        } else {
            eprintln!("  LoRA: {} (strength {lora_single})", lp.display());
        }
    }
    let lora_ckpt = |strength: f32| -> R<Option<LtxCheckpoint>> {
        match (&args.lora, strength > 0.0) {
            (Some(lp), true) => {
                let lw = LoraWeights::open(lp, dev, strength).map_err(|e| format!("LoRA {}: {e}", lp.display()))?;
                Ok(Some(ckpt.view_on(Device::Cpu).with_lora(std::sync::Arc::new(lw))))
            }
            _ => Ok(None),
        }
    };
    let build_avdit = |strength: f32| -> R<AvDit> {
        let lc = lora_ckpt(strength)?;
        Ok(AvDit::load_with(lc.as_ref().unwrap_or(&ckpt), dev, compute, quant_dit, offload)
            .map_err(|e| format!("AvDit(LoRA {strength}): {e}"))?)
    };
    let build_dit = |strength: f32| -> R<VideoDit> {
        let lc = lora_ckpt(strength)?;
        Ok(VideoDit::load_with(lc.as_ref().unwrap_or(&ckpt), dev, compute, quant_dit, offload)
            .map_err(|e| format!("VideoDit(LoRA {strength}): {e}"))?)
    };

    let t_enc = std::time::Instant::now();
    let enc_prof = args.prof;
    // Gemma 23GB + коннекторы шли pageable-H2D (~3.6GB/s): pinned-staging
    // конвейер (45GB/s) режет text-enc на ~5-8s. Async-копии упорядочены на
    // default stream (тот же стрим у потребителей).
    synaptix_core::device::cuda::set_offload_pinned(true);
    // Gemma-states считаются и Gemma ДРОПАЕТСЯ до загрузки коннекторов:
    // bf16-Gemma (24GB) иначе не помещается вместе с ними на 24GB-карте
    // (states 49×[1,1024,3840] ≈ 0.4GB — дёшево пережить дроп).
    let (states, mask, neg_states, nag_states) = {
        let prec = PrecisionConfig {
            compute,
            attn_w: quant_enc,
            mlp_w: quant_enc,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        };
        // Длина контекста 1024 = ОФИЦИАЛЬНАЯ (LTXVGemmaTokenizer(root, 1024)):
        // коннектор-перцивер тренирован на S=1024 (валидные + register-tile 8×128).
        // Прежние 128 системно смещали audio_encoding -> искажённая речь
        // (видео малочувствительно; вскрыто сверкой контекстов с эталоном).
        let gemma = GemmaPipeline::load_with_precision(&args.gemma, dev, prec, Some(1024))
            .map_err(|e| format!("Gemma load: {e}"))?;
        if enc_prof { eprintln!("[LTX_PROF] gemma-load: {:.1}s", t_enc.elapsed().as_secs_f32()); }
        // Квант-Gemma (резидентная, RAM-конфликта с зеркалом нет): резюмим
        // зеркалирование ckpt сразу — параллельно encode+коннекторам (как было:
        // text-enc 8.8s; отложенный resume гнал cuMemHostAlloc(44GB)+par_copy
        // внутрь окна text-encode → 20.3s). Dense-Gemma (host-stream, CPU-блоки
        // 21.5GB) — resume после дропа (RAM-OOM иначе).
        if quant_enc.is_quantized() {
            if let Some(g) = &_pin_cache {
                g.resume();
            }
        }
        let t_ge = std::time::Instant::now();
        let (states, mask) = gemma
            .encode_for_ltx(&args.prompt, 1024, dev)
            .map_err(|e| format!("Gemma encode: {e}"))?;
        if enc_prof { eprintln!("[LTX_PROF] gemma-encode: {:.1}s", t_ge.elapsed().as_secs_f32()); }
        let neg = if guided {
            let (ns, nm) = gemma
                .encode_for_ltx(&args.negative_prompt, 1024, dev)
                .map_err(|e| format!("Gemma encode neg: {e}"))?;
            Some((ns, nm))
        } else {
            None
        };
        // пустая строка = NAG выкл (--nag-prompt "")
        let nag_p = args.nag_prompt.as_deref().map(str::trim).filter(|s| !s.is_empty());
        let nag = match nag_p {
            Some(np) => {
                let (ns, nm) = gemma
                    .encode_for_ltx(np, 1024, dev)
                    .map_err(|e| format!("Gemma encode nag: {e}"))?;
                Some((ns, nm))
            }
            None => None,
        };
        let _t_drop = std::time::Instant::now();
        let r = (states, mask, neg, nag);
        if enc_prof { eprintln!("[LTX_PROF] pre-drop: {:.1}s", t_enc.elapsed().as_secs_f32()); }
        r
    }; // gemma освобождена (~12-24GB)
    if enc_prof { eprintln!("[LTX_PROF] post-drop: {:.1}s", t_enc.elapsed().as_secs_f32()); }
    // Зеркалирование ckpt возобновляется ПОСЛЕ дропа Gemma: при host-stream
    // bf16-энкодере CPU-блоки Gemma (21.5GB) + pinned-зеркало (44GB) вместе
    // выбивали RAM-лимит (OOM-kill 137).
    if let Some(g) = &_pin_cache {
        g.resume();
    }
    if let Device::Cuda(o) = dev {
        let _ = synaptix_core::device::cuda::synchronize_all(o);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(o);
    }
    if enc_prof { eprintln!("[LTX_PROF] post-trim: {:.1}s", t_enc.elapsed().as_secs_f32()); }
    let (v_enc, a_enc, neg_enc, nag_enc) = {
        // коннекторы грузят веса с ckpt.device → нужен Cuda-вью (ckpt открыт на Cpu
        // для DiT-offload; FeatureExtractorV2::load не принимает device-параметр).
        let ckpt_gpu = ckpt.view_on(dev);
        let vtc = VideoTextConditioner::load(&ckpt_gpu, dev, compute)?;
        let v = vtc.forward(&states, &mask)?;
        let a = if need_actx {
            Some(AudioTextConditioner::load(&ckpt_gpu, dev, compute)?.forward(&states, &mask)?)
        } else {
            None
        };
        // negative-context для CFG (тот же video-conditioner, negative_prompt).
        let neg = match &neg_states {
            Some((ns, nm)) => Some(vtc.forward(ns, nm)?),
            None => None,
        };
        let nag = match &nag_states {
            Some((ns, nm)) => Some(vtc.forward(ns, nm)?),
            None => None,
        };
        (v, a, neg, nag)
    }; // states + коннекторы освобождены
    drop(states);
    drop(neg_states);
    drop(nag_states);
    let v_nag = nag_enc.as_ref().map(|t| (t, args.nag_scale, args.nag_alpha, args.nag_tau));
    if v_nag.is_some() {
        eprintln!("  NAG: scale={} alpha={} tau={} (stage1 v_attn2)", args.nag_scale, args.nag_alpha, args.nag_tau);
    }
    synaptix_core::device::cuda::set_offload_pinned(false);
    eprintln!("  text-encoding: {:.1}s (v {:?})", t_enc.elapsed().as_secs_f32(), v_enc.dims());


    let t_gen = std::time::Instant::now();
    let (latent, a_tok, input_wave) = synaptix_core::grad::no_grad(|| -> R<(Tensor, Option<Tensor>, Option<Tensor>)> {
        if !av_path {
            return Ok((generate_video_only(&args, &ckpt, &v_enc, neg_enc.as_ref(), &guider, &build_dit, two_stage, refine, fp, hp, wp, dev, compute, seed, (lora_single, lora_s1, lora_s2))?, None, None));
        }
        let a_enc_t = a_enc.as_ref().expect("аудио-контекст считается для A/V-пути");
        let p1 = progress_printer("stage1");
        let p2 = progress_printer("stage2");
        let hooks1 = DenoiseHooks { progress: Some(&p1), cancel: None };
        let hooks2 = DenoiseHooks { progress: Some(&p2), cancel: None };
        let (hp1, wp1) = if two_stage { stage1_grid(hp, wp) } else { (hp, wp) };
        let (hp2, wp2) = (hp1 * 2, wp1 * 2);
        let (e1, e2) = if two_stage { (lora_s1, lora_s2) } else { (lora_single, lora_single) };
        let mut dit = Some(build_avdit(e1)?);
        let mut up = if two_stage {
            Some(load_upsampler(&ckpt, args.upscaler.as_ref().expect("проверено выше"), dev)?)
        } else {
            None
        };
        let stage2_dit = |dit: &mut Option<AvDit>| -> R<()> {
            if e1 != e2 {
                *dit = None;
                release_vram(dev);
                *dit = Some(build_avdit(e2)?);
            }
            Ok(())
        };
        let upsample = |l1: &Tensor, up: &mut Option<Upsampler>| -> R<Tensor> {
            let l2 = up.as_ref().expect("upscaler").upsample(l1)?.to_dtype(compute)?;
            *up = None;
            synaptix_core::tensor::ops::conv_filter_cache_clear();
            Ok(l2)
        };
        let venc = if matches!(mode, Mode::Text | Mode::A2v { .. }) {
            None
        } else {
            Some(VaeEncoder::load(&ckpt, dev).map_err(|e| format!("VAE encoder: {e}"))?)
        };
        let result = match &mode {
            Mode::Text => {
                let (l1, a1) = denoise_av(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, None, args.fps, dev, v_nag, &[], seed, &hooks1)?;
                if two_stage {
                    let l2 = upsample(&l1, &mut up)?;
                    if refine {
                        stage2_dit(&mut dit)?;
                        let (l, a) = denoise_av(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp2, wp2, &STAGE2_SIGMAS, Some(&l2), Some(&a1), args.fps, dev, None, &[], seed, &hooks2)?;
                        (l, Some(a), None)
                    } else {
                        (l2, Some(a1), None)
                    }
                } else {
                    (l1, Some(a1), None)
                }
            }
            Mode::Image { path, strength, frame } => {
                let raw = synaptix_io::image::load_image(path, Device::Cpu).map_err(|e| format!("image {e}"))?;
                let encoder = venc.as_ref().unwrap();
                let run_stage = |dit: &AvDit, g: (usize, usize), sigmas: &[f64], init: Option<(&Tensor, &Tensor)>, nag: Option<(&Tensor, f32, f32, f32)>, hooks: &DenoiseHooks| -> R<(Tensor, Tensor)> {
                    let toks = image_tokens(encoder, &raw, g.0, g.1, dev)?;
                    let (vi, ai) = match init { Some((v, a)) => (Some(v), Some(a)), None => (None, None) };
                    Ok(if *frame > 0 {
                        let pos = keyframe_positions(g.0, g.1, *frame, args.fps);
                        denoise_av_append(dit, &v_enc, a_enc_t, fp, g.0, g.1, sigmas, vi, ai, false, Some((&toks, &pos, *strength)), None, args.fps, dev, seed, hooks)?
                    } else {
                        denoise_av(dit, &v_enc, a_enc_t, fp, g.0, g.1, sigmas, vi, ai, args.fps, dev, nag, &[(0, toks, *strength)], seed, hooks)?
                    })
                };
                eprintln!("  {}: {} (strength {strength}{})", mode.label(), path.display(), if *frame > 0 { format!(", кадр {frame}") } else { String::new() });
                let (l1, a1) = run_stage(dit.as_ref().unwrap(), (hp1, wp1), &DISTILLED_SIGMAS, None, v_nag, &hooks1)?;
                if two_stage {
                    let l2 = upsample(&l1, &mut up)?;
                    if refine {
                        stage2_dit(&mut dit)?;
                        let (l, a) = run_stage(dit.as_ref().unwrap(), (hp2, wp2), &STAGE2_SIGMAS, Some((&l2, &a1)), None, &hooks2)?;
                        (l, Some(a), None)
                    } else {
                        (l2, Some(a1), None)
                    }
                } else {
                    (l1, Some(a1), None)
                }
            }
            Mode::Retake { path, start, end } => {
                let frames = load_video_frames(path, hp * 32, wp * 32, out_frames, args.fps, dev)?;
                let source = venc.as_ref().unwrap().encode(&frames).map_err(|e| format!("VAE encode: {e}"))?;
                let (fps_, hps, wps) = (source.dims()[2], source.dims()[3], source.dims()[4]);
                eprintln!("  retake: {} регион [{start:.2}, {end:.2}] с, латент {fps_}×{hps}×{wps}", path.display());
                let (l, a) = denoise_av_retake(dit.as_ref().unwrap(), &v_enc, a_enc_t, fps_, hps, wps, &DISTILLED_SIGMAS, &source, *start, *end, args.fps, dev, seed, &hooks1)?;
                (l, Some(a), None)
            }
            Mode::IcLora { path } => {
                let ds = args.ref_downscale.max(1);
                let encoder = venc.as_ref().unwrap();
                let make_ref = |g: (usize, usize)| -> R<(Tensor, Vec<f64>)> {
                    let (rph, rpw) = (g.0 * 32 / ds, g.1 * 32 / ds);
                    let mut frames = load_video_frames(path, rph, rpw, out_frames, args.fps, dev)?;
                    frames = match args.ref_preprocess.as_str() {
                        "canny" => apply_canny_frames(&frames, args.canny_low, args.canny_high, args.control_preview.as_deref())?,
                        "depth" => apply_depth_frames(&frames, &args.depth_model, dev, args.control_preview.as_deref())?,
                        "none" => frames,
                        o => return Err(format!("--ref-preprocess: none | canny | depth, а не `{o}`").into()),
                    };
                    let lat = encoder.encode(&frames).map_err(|e| format!("ref encode: {e}"))?;
                    let (tok, fpr, hr, wr) = latent_tokens(&lat)?;
                    Ok((tok, ref_video_positions(fpr, hr, wr, args.fps, ds)))
                };
                eprintln!("  IC-LoRA ref: {} (downscale {ds}, strength {}, control {})", path.display(), args.ref_strength, args.ref_preprocess);
                let (r1, p1pos) = make_ref((hp1, wp1))?;
                let (l1, a1) = denoise_av_append(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, None, false, Some((&r1, &p1pos, args.ref_strength)), None, args.fps, dev, seed, &hooks1)?;
                if two_stage {
                    let l2 = upsample(&l1, &mut up)?;
                    if refine {
                        stage2_dit(&mut dit)?;
                        let (r2, p2pos) = make_ref((hp2, wp2))?;
                        let (l, a) = denoise_av_append(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp2, wp2, &STAGE2_SIGMAS, Some(&l2), Some(&a1), false, Some((&r2, &p2pos, args.ref_strength)), None, args.fps, dev, seed, &hooks2)?;
                        (l, Some(a), None)
                    } else {
                        (l2, Some(a1), None)
                    }
                } else {
                    (l1, Some(a1), None)
                }
            }
            Mode::A2v { audio } => {
                let fa = audio_token_count(fp, args.fps);
                let audio_lat = audio_input_latent(&ckpt, audio, fa, dev)?;
                eprintln!("  audio→video: {} ({fa} аудио-токенов, заморожено)", audio.display());
                let (l1, _) = denoise_av_append(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, Some(&audio_lat), true, None, None, args.fps, dev, seed, &hooks1)?;
                let latent = if two_stage {
                    let l2 = upsample(&l1, &mut up)?;
                    if refine {
                        stage2_dit(&mut dit)?;
                        denoise_av_append(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp2, wp2, &STAGE2_SIGMAS, Some(&l2), Some(&audio_lat), true, None, None, args.fps, dev, seed, &hooks2)?.0
                    } else {
                        l2
                    }
                } else {
                    l1
                };
                (latent, None, Some(load_audio_48k_stereo(audio)?))
            }
            Mode::Lipdub { reference, audio } => {
                let encoder = venc.as_ref().unwrap();
                let make_ref = |g: (usize, usize)| -> R<(Tensor, Vec<f64>)> {
                    let frames = load_video_frames(reference, g.0 * 32, g.1 * 32, out_frames, args.fps, dev)?;
                    let lat = encoder.encode(&frames).map_err(|e| format!("ref encode: {e}"))?;
                    let (tok, fpr, hr, wr) = latent_tokens(&lat)?;
                    Ok((tok, synaptix_video_ltx23::pipeline::pixel_coords(fpr, hr, wr, args.fps)))
                };
                let aenc = AudioVaeEncoder::load(&ckpt, dev).map_err(|e| format!("audio encoder: {e}"))?;
                let mel = ltx_log_mel(&[load_audio_16k(audio)?], dev).map_err(|e| format!("mel: {e}"))?;
                let audio_ref = aenc.encode(&mel).map_err(|e| format!("audio encode: {e}"))?;
                drop(aenc);
                eprintln!("  lipdub: ref={} audio={} ({} ток)", reference.display(), audio.display(), audio_ref.dims()[1]);
                let (r1, r1pos) = make_ref((hp1, wp1))?;
                let a_ref_pos = lipdub_audio_ref_positions(audio_ref.dims()[1]);
                let (l1, a1) = denoise_av_append(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, None, false, Some((&r1, &r1pos, args.ref_strength)), Some((&audio_ref, &a_ref_pos)), args.fps, dev, seed, &hooks1)?;
                let l2 = upsample(&l1, &mut up)?;
                let latent = if refine {
                    stage2_dit(&mut dit)?;
                    let (r2, r2pos) = make_ref((hp2, wp2))?;
                    let a1_pos = lipdub_audio_ref_positions(a1.dims()[1]);
                    denoise_av_append(dit.as_ref().unwrap(), &v_enc, a_enc_t, fp, hp2, wp2, &STAGE2_SIGMAS, Some(&l2), Some(&a1), true, Some((&r2, &r2pos, args.ref_strength)), Some((&a1, &a1_pos)), args.fps, dev, seed, &hooks2)?.0
                } else {
                    l2
                };
                (latent, None, Some(load_audio_48k_stereo(audio)?))
            }
        };
        drop(dit);
        drop(up);
        drop(venc);
        synaptix_core::tensor::ops::conv_filter_cache_gc();
        Ok(result)
    })?;
    let _ = synaptix_core::device::cuda::synchronize(ord);
    eprintln!("  денойз: {:.1}s", t_gen.elapsed().as_secs_f32());
    release_vram(dev);
    let tv = std::time::Instant::now();
    let rgb = synaptix_core::grad::no_grad(|| -> R<Tensor> {
        let vae = VaeDecoder::load(&ckpt, dev).map_err(|e| format!("VAE: {e}"))?;
        Ok(vae.decode(&latent).map_err(|e| format!("VAE decode: {e}"))?)
    })?;
    let _ = synaptix_core::device::cuda::synchronize(ord);
    eprintln!("  vae-decode: {:.1}s → {} кадров {}×{}", tv.elapsed().as_secs_f32(), rgb.dims()[2], rgb.dims()[3], rgb.dims()[4]);
    let wave = if !keep_audio {
        None
    } else if let Some(w) = input_wave {
        Some(w)
    } else if let Some(a) = &a_tok {
        let ta = std::time::Instant::now();
        let wave = synaptix_core::grad::no_grad(|| -> R<Tensor> {
            let audio_vae = AudioVaeDecoder::load(&ckpt, dev).map_err(|e| format!("audio VAE: {e}"))?;
            let vocoder = VocoderWithBwe::load(&args.model, dev).map_err(|e| format!("vocoder: {e}"))?;
            Ok(decode_audio_tokens(&audio_vae, &vocoder, a)?)
        })?;
        eprintln!("  audio-decode+vocoder: {:.1}s", ta.elapsed().as_secs_f32());
        Some(wave)
    } else {
        None
    };
    write_mp4(&rgb, wave.as_ref(), args.fps, args.crf, &args.output)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn generate_video_only(
    args: &VideoArgs,
    ckpt: &LtxCheckpoint,
    v_enc: &Tensor,
    neg_enc: Option<&Tensor>,
    guider: &GuiderParams,
    build_dit: &dyn Fn(f32) -> R<VideoDit>,
    two_stage: bool,
    refine: bool,
    fp: usize,
    hp: usize,
    wp: usize,
    dev: Device,
    compute: DType,
    seed: Option<u64>,
    lora: (f32, f32, f32),
) -> R<Tensor> {
    let (single, s1, s2) = lora;
    let ctx_t = v_enc.to_device(dev)?.to_dtype(compute)?;
    let p1 = progress_printer("stage1");
    let p2 = progress_printer("stage2");
    let hooks1 = DenoiseHooks { progress: Some(&p1), cancel: None };
    let hooks2 = DenoiseHooks { progress: Some(&p2), cancel: None };
    if !two_stage {
        let dit = build_dit(single)?;
        return Ok(denoise(&dit, &ctx_t, fp, hp, wp, &DISTILLED_SIGMAS, None, args.fps, dev, seed, &hooks1)?);
    }
    let (hp1, wp1) = stage1_grid(hp, wp);
    let up = load_upsampler(ckpt, args.upscaler.as_ref().expect("проверено выше"), dev)?;
    let guided = neg_enc.is_some() && (guider.cfg_scale > 1.0 || guider.stg_scale > 0.0);
    let (e1, e2) = if guided { (0.0, s2) } else { (s1, s2) };
    let mut dit = build_dit(e1)?;
    let l1 = if guided {
        let neg_t = neg_enc.expect("neg-context при guided").to_device(dev)?.to_dtype(compute)?;
        let sg = synaptix_video_ltx23::pipeline::ltx2_sigmas(args.steps, fp * hp1 * wp1);
        synaptix_video_ltx23::pipeline::denoise_video_guided(&dit, &ctx_t, &neg_t, guider, fp, hp1, wp1, &sg, None, args.fps, dev, seed, &hooks1)?
    } else {
        denoise(&dit, &ctx_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, args.fps, dev, seed, &hooks1)?
    };
    let l2 = up.upsample(&l1)?.to_dtype(compute)?;
    drop(up);
    synaptix_core::tensor::ops::conv_filter_cache_clear();
    if !refine {
        return Ok(l2);
    }
    if e1 != e2 {
        drop(dit);
        release_vram(dev);
        dit = build_dit(e2)?;
    }
    Ok(denoise(&dit, &ctx_t, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS, Some(&l2), args.fps, dev, seed, &hooks2)?)
}

/// Аудиофайл → волна `[1,2,L]` 48kHz stereo f32 (для мукса в mp4 через write_mp4).
fn load_audio_48k_stereo(path: &std::path::Path) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let out = Command::new("ffmpeg")
        .args(["-y", "-i"]).arg(path)
        .args(["-ar", "48000", "-ac", "2", "-f", "f32le", "-"])
        .output()?;
    if !out.status.success() {
        return Err(format!("ffmpeg audio 48k {path:?}: {}", out.status).into());
    }
    let inter: Vec<f32> = out.stdout.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    let l = inter.len() / 2;
    // interleaved LR → planar [1,2,L]
    let mut planar = vec![0f32; 2 * l];
    for i in 0..l {
        planar[i] = inter[2 * i];
        planar[l + i] = inter[2 * i + 1];
    }
    Ok(synaptix_core::tensor::Tensor::from_vec(planar, vec![1, 2, l], Device::Cpu)?)
}

/// Depth Anything V2 по всем кадрам `[1,3,F,H,W]` ([−1,1]) → карты глубины той же
/// формы (ближе=белее, [−1,1]).
fn apply_depth_frames(
    frames: &synaptix_core::tensor::Tensor,
    model_dir: &std::path::Path,
    dev: Device,
    preview: Option<&std::path::Path>,
) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let m = synaptix_depth_anything::DepthAnything::load(model_dir, dev)
        .map_err(|e| format!("depth model: {e}"))?;
    let (f, h, w) = (frames.dims()[2], frames.dims()[3], frames.dims()[4]);
    let mut out: Vec<synaptix_core::tensor::Tensor> = Vec::with_capacity(f);
    for fi in 0..f {
        let fr = frames.narrow(2, fi, 1)?.contiguous()?.reshape(vec![3, h, w])?
            .affine(0.5, 0.5)?; // [−1,1] → [0,1]
        let d = m.depth_rgb(&fr).map_err(|e| format!("depth: {e}"))?;
        if let (0, Some(p)) = (fi, preview) {
            synaptix_io::image::save_image(&d, p)?;
        }
        out.push(d.affine(2.0, -1.0)?.reshape(vec![1, 3, 1, h, w])?);
    }
    let refs: Vec<&synaptix_core::tensor::Tensor> = out.iter().collect();
    Ok(synaptix_core::tensor::Tensor::cat(&refs, 2)?.contiguous()?)
}

/// Canny по всем кадрам `[1,3,F,H,W]` ([−1,1]) → контурный control-сигнал той же
/// формы (белые рёбра на чёрном, [−1,1]).
fn apply_canny_frames(
    frames: &synaptix_core::tensor::Tensor,
    low: f32,
    high: f32,
    preview: Option<&std::path::Path>,
) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let (f, h, w) = (frames.dims()[2], frames.dims()[3], frames.dims()[4]);
    let mut out: Vec<synaptix_core::tensor::Tensor> = Vec::with_capacity(f);
    for fi in 0..f {
        let fr = frames.narrow(2, fi, 1)?.contiguous()?.reshape(vec![3, h, w])?
            .affine(0.5, 0.5)?; // [−1,1] → [0,1]
        let edges = synaptix_io::image::canny_rgb(&fr, low, high).map_err(|e| format!("canny: {e}"))?;
        if let (0, Some(p)) = (fi, preview) {
            synaptix_io::image::save_image(&edges, p)?;
        }
        out.push(edges.affine(2.0, -1.0)?.reshape(vec![1, 3, 1, h, w])?);
    }
    let refs: Vec<&synaptix_core::tensor::Tensor> = out.iter().collect();
    Ok(synaptix_core::tensor::Tensor::cat(&refs, 2)?.contiguous()?)
}

/// Аудиофайл → 16kHz mono f32-сэмплы (ffmpeg: -ar 16000 -ac 1 -f f32le).
fn load_audio_16k(path: &std::path::Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let out = Command::new("ffmpeg")
        .args(["-y", "-i"]).arg(path)
        .args(["-ar", "16000", "-ac", "1", "-f", "f32le", "-"])
        .output()?;
    if !out.status.success() {
        return Err(format!("ffmpeg audio decode {path:?}: {}", out.status).into());
    }
    Ok(out.stdout.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

fn scratch_dir(tag: &str) -> std::io::Result<PathBuf> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("synaptix_{tag}_{}_{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

fn load_video_frames(
    path: &std::path::Path,
    ph: usize,
    pw: usize,
    n: usize,
    fps: f64,
    dev: Device,
) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let dir = scratch_dir("vin")?;
    let result = (|| -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(path)
            .args(["-vf", &format!("fps={fps},scale={pw}:{ph}")])
            .args(["-frames:v", &n.to_string()])
            .arg(dir.join("f%05d.png"))
            .status()?;
        if !status.success() {
            return Err(format!("ffmpeg decode {path:?} → {status}").into());
        }
        let mut frames: Vec<synaptix_core::tensor::Tensor> = Vec::with_capacity(n);
        for i in 1..=n {
            let p = dir.join(format!("f{i:05}.png"));
            if !p.exists() {
                break;
            }
            let img = synaptix_io::image::load_image(&p, dev).map_err(|e| format!("frame {i}: {e}"))?;
            frames.push(img.contiguous()?.affine(2.0, -1.0)?.contiguous()?.reshape(vec![1, 3, 1, ph, pw])?);
        }
        if frames.is_empty() {
            return Err(format!("{}: видео не дало кадров", path.display()).into());
        }
        let refs: Vec<&synaptix_core::tensor::Tensor> = frames.iter().collect();
        Ok(synaptix_core::tensor::Tensor::cat(&refs, 2)?.contiguous()?)
    })();
    let _ = std::fs::remove_dir_all(&dir);
    result
}

/// RGB `[1,3,F,H,W]` (+опц. стерео wave `[1,2,L]`) → mp4 через ffmpeg (PPM-кадры
/// + WAV в temp-директории).
fn write_mp4(
    rgb: &synaptix_core::tensor::Tensor,
    wave: Option<&synaptix_core::tensor::Tensor>,
    fps: f64,
    crf: Option<u32>,
    out: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = scratch_dir("vout")?;
    let frames = rgb_to_frames(rgb)?;
    for (i, fr) in frames.iter().enumerate() {
        let (h, w) = (fr.dims()[1], fr.dims()[2]);
        let planar: Vec<f32> = fr.reshape(vec![3 * h * w])?.to_vec1::<f32>()?;
        let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    buf.push((planar[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0).round() as u8);
                }
            }
        }
        std::fs::write(dir.join(format!("f{i:05}.ppm")), buf)?;
    }
    let frame_glob = dir.join("f%05d.ppm");
    let fr_s = format!("{fps}");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-v", "error", "-y", "-framerate", &fr_s, "-i"]).arg(&frame_glob);
    let wav_path = dir.join("audio.wav");
    if let Some(wave) = wave {
        write_wav(&wav_path, wave, 48000)?;
        cmd.arg("-i").arg(&wav_path).args(["-c:a", "aac", "-shortest"]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]);
    if let Some(crf) = crf {
        cmd.args(["-crf", &crf.to_string()]);
    }
    cmd.arg(out);
    let status = cmd.status()?;
    if status.success() {
        let _ = std::fs::remove_dir_all(&dir);
        eprintln!("WROTE {} ({} кадров{})", out.display(), frames.len(),
            if wave.is_some() { " + аудио" } else { "" });
    } else {
        return Err(format!("ffmpeg завершился с {status} (PPM в {})", dir.display()).into());
    }
    Ok(())
}

/// Стерео wave `[1,2,L]` f32 [-1,1] → 16-бит PCM WAV.
fn write_wav(
    path: &std::path::Path,
    wave: &synaptix_core::tensor::Tensor,
    sr: u32,
) -> Result<(), Box<dyn std::error::Error>> {
    let (ch, l) = (wave.dims()[1], wave.dims()[2]);
    let data: Vec<f32> = wave.reshape(vec![ch * l])?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    let mut pcm: Vec<u8> = Vec::with_capacity(ch * l * 2);
    for i in 0..l {
        for c in 0..ch {
            let s = (data[c * l + i].clamp(-1.0, 1.0) * 32767.0) as i16;
            pcm.extend_from_slice(&s.to_le_bytes());
        }
    }
    let byte_rate = sr * ch as u32 * 2;
    let block_align = (ch * 2) as u16;
    let data_len = pcm.len() as u32;
    let mut f: Vec<u8> = Vec::new();
    f.extend_from_slice(b"RIFF");
    f.extend_from_slice(&(36 + data_len).to_le_bytes());
    f.extend_from_slice(b"WAVEfmt ");
    f.extend_from_slice(&16u32.to_le_bytes());
    f.extend_from_slice(&1u16.to_le_bytes()); // PCM
    f.extend_from_slice(&(ch as u16).to_le_bytes());
    f.extend_from_slice(&sr.to_le_bytes());
    f.extend_from_slice(&byte_rate.to_le_bytes());
    f.extend_from_slice(&block_align.to_le_bytes());
    f.extend_from_slice(&16u16.to_le_bytes()); // bits
    f.extend_from_slice(b"data");
    f.extend_from_slice(&data_len.to_le_bytes());
    f.extend_from_slice(&pcm);
    std::fs::write(path, f)?;
    Ok(())
}

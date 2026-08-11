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
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_video_ltx23::dit::{AvDit, VideoDit};
use synaptix_video_ltx23::loader::{LoraWeights, LtxCheckpoint};
use synaptix_video_ltx23::guider::GuiderParams;
use synaptix_video_ltx23::audio_vae::{ltx_log_mel, AudioVaeEncoder};
use synaptix_video_ltx23::pipeline::{
    decode_audio_tokens, denoise, denoise_av, fp_for_frames, frames_for_duration, generate_av,
    generate_ic_lora_video, generate_image_to_video, generate_image_to_video_two_stage,
    generate_keyframe_to_video, generate_lipdub_latents, generate_retake_video, generate_video,
    latent_grid, out_frame_count, pixel_coords, rgb_to_frames, stage1_grid, DenoiseHooks,
    DISTILLED_SIGMAS, STAGE2_SIGMAS, SUPPORTED_FPS,
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
    pub lora_strength_stage1: f32, // two-stage: сила LoRA на stage1 (офиц. HQ 0.25)
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

fn parse_quant(s: Option<&str>, default: DType) -> Result<DType, String> {
    match s.map(|x| x.to_lowercase()) {
        None => Ok(default),
        Some(q) => match q.as_str() {
            "none" | "bf16" => Ok(DType::BF16),
            "f16" => Ok(DType::F16),
            "nvfp4" => Ok(DType::NVFP4),
            "mxfp8" | "fp8" => Ok(DType::MXFP8),
            other => Err(format!("неизвестный квант: {other} (none|mxfp8|nvfp4)")),
        },
    }
}

pub fn run(args: VideoArgs) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_video_ltx23::spec;
    if args.prof {
        synaptix_video_ltx23::runtime::set_ltx_prof(true);
    }
    if let Some(m) = args.block_mode {
        synaptix_video_ltx23::runtime::set_ltx_block_mode(m);
    }
    // --list-pipelines: печать реестра и выход (без загрузки моделей).
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

    let dev = match args.device.as_str() {
        "cpu" => Device::Cpu,
        _ => Device::Cuda(0),
    };
    let compute = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(o) => return Err(format!("unknown compute-dtype {o}").into()),
        None => DType::BF16,
    };
    // квант блоков DiT: dense bf16 → streaming-offload; квант → резидентно если
    // влезает в VRAM, иначе host-stream offload (решение ниже, после открытия ckpt)
    let quant_dit = parse_quant(args.quant_transformer.as_deref(), compute)?;
    // квант весов Gemma (дефолт mxfp8 — 12B влезает в ~12GB)
    let quant_enc = parse_quant(args.quant_encoder.as_deref(), DType::MXFP8)?;

    if !SUPPORTED_FPS.contains(&args.fps) {
        return Err(format!("--fps {} не поддерживается (24 | 25 | 48 | 50)", args.fps).into());
    }
    // сетка латента: H=32·hp, W=32·wp; F=8·(fp−1)+1
    let (hp, wp) = latent_grid(args.width, args.height);
    // кадры: явный --frames (floor к сетке, back-compat) либо --duration·fps
    // (round к ближайшему 8·(fp−1)+1, чтобы длительность держалась точнее)
    let fp = match args.frames {
        Some(f) => fp_for_frames(f),
        None => frames_for_duration(parse_duration_secs(&args.duration)?, args.fps),
    };
    let out_frames = out_frame_count(fp);
    // Резолв пайплайна: явный --pipeline переопределяет; иначе выводим из флагов
    // (back-compat: --two-stage → two-stage, --no-audio → one-stage, иначе av).
    let sp = match &args.pipeline {
        Some(name) => spec::by_name(name).ok_or_else(|| {
            format!("неизвестный --pipeline {name} (доступно: {})", spec::names().join(", "))
        })?,
        None => {
            let derived = if args.two_stage { "two-stage" } else if args.no_audio { "one-stage" } else { "av" };
            spec::by_name(derived).expect("derived pipeline known")
        }
    };
    if !sp.implemented() {
        return Err(format!(
            "пайплайн «{}» ещё не реализован (Фаза {}). Готовы: {}",
            sp.name, sp.todo_phase.unwrap(),
            spec::registry().iter().filter(|p| p.implemented()).map(|p| p.name).collect::<Vec<_>>().join(", "),
        ).into());
    }
    let two_stage = sp.stages == spec::Stages::Two;
    let mut audio = sp.modality == spec::Modality::AudioVideo && !args.no_audio;
    let refine = sp.refine && !args.no_refine;
    // multimodal guidance: cfg>1 или stg>0 (Фаза 3, для не-distilled чекпойнта).
    // Реализован guided video-путь на two-stage (guided stage1 → distilled refine).
    let guided = args.cfg_scale > 1.0 || args.stg_scale > 0.0;
    if guided && !two_stage {
        return Err("guidance (--cfg-scale/--stg-scale) поддержан только на two-stage (guided stage1)".into());
    }
    if guided && audio {
        eprintln!("  [внимание] guided-путь сейчас только видео → аудио отключено");
        audio = false;
    }
    if sp.needs_upscaler() && args.upscaler.is_none() {
        return Err(format!("пайплайн «{}» требует --upscaler <spatial-upscaler.safetensors>", sp.name).into());
    }
    if sp.needs_lora && args.lora.is_none() {
        eprintln!("  [внимание] пайплайн «{}» рассчитан на IC-LoRA — укажи --lora", sp.name);
    }
    let guider = GuiderParams {
        cfg_scale: args.cfg_scale,
        stg_scale: args.stg_scale,
        rescale_scale: if guided { 0.7 } else { 0.0 },
        modality_scale: 1.0, // isolated-modality пока не задействован (video-only)
        skip_step: 0,
        stg_blocks: vec![29],
    };
    let mut ckpt = LtxCheckpoint::open(&args.model, Device::Cpu, DType::BF16)
        .map_err(|e| format!("LTX ckpt: {e}"))?;
    // Авто-решение резидент-vs-offload для квантованного DiT: веса (по факту из
    // ckpt) + VAE + пиковые активации блока должны влезать в свободную VRAM, иначе
    // host-stream offload (блоки квантуются 1× → host-RAM → стримятся на GPU).
    let offload = if quant_dit == compute {
        true // dense → стримим mmap→VRAM (резидентно 43GB не влезает заведомо)
    } else if args.force_offload {
        true
    } else if let Device::Cuda(ord) = dev {
        let dit_b = synaptix_video_ltx23::dit::dit_resident_bytes(&ckpt, quant_dit, compute);
        let vae_b: usize = ckpt
            .infos()
            .filter(|(n, _, _)| n.starts_with("vae."))
            .map(|(_, _, s)| compute.bytes_for_numel(s.iter().product()))
            .sum();
        let act_b = fp * hp * wp * 4096 * 2 * 28; // modul[9·dim]+rope+ff-промежуточные, bf16
        // Квант-путь паддит M до 256 и держит pад-буферы (x_pad f16 m·k_max,
        // out_pad m·n_max f16, packed/scales): на FullHD ~2-3GB поверх act_b —
        // без учёта nvfp4 ложно влезал резидентно и падал OOM на stage2 (2026-06-05).
        let quant_pad_b = (fp * hp * wp + 256) * (16384 * 2 + 16384 * 2 + 12288 * 2);
        let need = dit_b + vae_b + act_b + quant_pad_b + (1usize << 30);
        let (free, _total) = synaptix_core::device::cuda::mem_info(ord)
            .map_err(|e| format!("mem_info: {e}"))?;
        if need > free {
            eprintln!(
                "  DiT {quant_dit:?} резидентно ~{:.1}GB (+VAE/активации) > свободно {:.1}GB → streaming-offload из host-RAM",
                dit_b as f64 / 1e9, free as f64 / 1e9,
            );
            true
        } else {
            false
        }
    } else {
        false
    };
    eprintln!(
        "synaptix video [{}]: «{}»\n  {}×{} (сетка {hp}×{wp}) | {out_frames} кадров @ {}fps (~{:.1}s) | аудио={audio} | two_stage={two_stage} refine={refine} | guided={guided}{}\n  DiT quant={quant_dit:?} (offload={offload}) | Gemma quant={quant_enc:?} | compute={compute:?} | {dev:?}",
        sp.name, args.prompt, wp * 32, hp * 32, args.fps, out_frames as f64 / args.fps,
        if guided { format!(" (cfg {} stg {} steps {})", args.cfg_scale, args.stg_scale, args.steps) } else { String::new() },
    );
    // dense-offload: фоновое зеркалирование ckpt в pinned host-RAM — к denoise
    // стрим блоков идёт из pinned (~45GB/s) вместо NVMe-перечиток (page cache не
    // держит 46GB цикла). Старт НА ПАУЗЕ: NVMe сперва нужен Gemma-load (конкуренция
    // давала +2.5s text-enc), резюм сразу после загрузки энкодера.
    let _pin_cache = if offload && quant_dit == compute && matches!(dev, Device::Cuda(_)) {
        Some(if quant_enc.is_quantized() {
            // квант-Gemma: pinned-зеркало аллоцируется сразу (вне text-enc окна)
            synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused(&ckpt.shard_bytes())
        } else {
            // dense-Gemma host-stream: лениво (44GB pinned + 21.5GB CPU-блоки = RAM-OOM)
            synaptix_core::device::cuda::OffloadPinCacheGuard::new_paused_lazy(&ckpt.shard_bytes())
        })
    } else {
        None
    };

    // ── 1. Живой Gemma → (video_encoding, audio_encoding), затем DROP Gemma ──
    // distilled-LoRA: мерджится в веса DiT при загрузке (W += strength·B@A). LoRA
    // открывается на compute-устройстве — дельты считаются там же, где материализуются веса.
    // ⚠ ОФИЦИАЛЬНАЯ семантика (args.py): distilled-чекпойнт LoRA НЕ использует вовсе;
    // dev ti2v — ТОЛЬКО stage2 (~0.8); dev HQ — stage1 0.25 / stage2 0.5. Strength 1.0
    // на обе стадии = накапливающаяся деградация на длинных видео (выяснено бисектом).
    // two-stage (guided и нет): per-stage strengths, мердж в ветке генерации.
    // Одностадийные пути: мердж в ckpt со strength --lora-strength.
    let lora_s2 = args.lora_strength_stage2.unwrap_or(args.lora_strength);
    if let Some(lora_path) = &args.lora {
        if two_stage {
            eprintln!("  LoRA: {} (stage1 {} / stage2 {})",
                lora_path.display(), args.lora_strength_stage1, lora_s2);
        } else {
            let lw = std::sync::Arc::new(LoraWeights::open(lora_path, dev, args.lora_strength)
                .map_err(|e| format!("LoRA {}: {e}", lora_path.display()))?);
            ckpt = ckpt.with_lora(lw);
            eprintln!("  LoRA: {} (strength {})", lora_path.display(), args.lora_strength);
        }
    }
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
        let a = if audio {
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

    // ── 2. DiT + VAE (+аудио) → RGB (+wave) ──
    let vae = VaeDecoder::load(&ckpt, dev).map_err(|e| format!("VAE: {e}"))?;
    let t_gen = std::time::Instant::now();
    let (rgb, wave) = synaptix_core::grad::no_grad(|| -> Result<_, Box<dyn std::error::Error>> {
        if let (Some(ref_path), Some(audio_path)) = (&args.ref_video, &args.audio) {
            // LIPDUB: лицо (ref-видео) + речь (wav) → видео с губами под аудио.
            // Официальный поток: lipdub-LoRA на обе стадии; stage1 joint A/V
            // (ref-видео append + ref-аудио append с отриц. позициями) → upscale →
            // stage2 видео-refine (ref full-res; аудио заморожено). Выход = видео
            // + сгенерированное аудио (≈входная речь).
            let a_enc_t = a_enc.as_ref().ok_or("lipdub требует аудио-контекст (без --no-audio)")?;
            let up_path = args.upscaler.as_ref().ok_or("lipdub требует --upscaler")?;
            let ml = SafetensorsLoader::open(&args.model).map_err(|e| format!("stats {e}"))?.with_device(dev);
            let mean = ml.load("vae.per_channel_statistics.mean-of-means").map_err(|e| format!("vae mean: {e}"))?;
            let std = ml.load("vae.per_channel_statistics.std-of-means").map_err(|e| format!("vae std: {e}"))?;
            let up = Upsampler::load(up_path, &mean, &std, dev).map_err(|e| format!("upscaler: {e}"))?;
            let (hp1, wp1) = stage1_grid(hp, wp);
            // ref-видео на разрешениях обеих стадий → VAE encode → токены+позиции
            let venc = VaeEncoder::load(&ckpt, dev).map_err(|e| format!("VAE encoder: {e}"))?;
            let mk_ref = |hpx: usize, wpx: usize| -> Result<(synaptix_core::tensor::Tensor, Vec<f64>), Box<dyn std::error::Error>> {
                let frames_t = load_video_frames(ref_path, hpx * 32, wpx * 32, out_frames, dev)?;
                let lat = venc.encode(&frames_t).map_err(|e| format!("ref encode: {e}"))?;
                let (fpr, hr, wr) = (lat.dims()[2], lat.dims()[3], lat.dims()[4]);
                let tok = lat.reshape(vec![1, 128, fpr * hr * wr])?.transpose(1, 2)?.contiguous()?;
                Ok((tok, pixel_coords(fpr, hr, wr, args.fps)))
            };
            let (ref1, ref1_pos) = mk_ref(hp1, wp1)?;
            let (ref2, ref2_pos) = mk_ref(hp1 * 2, wp1 * 2)?;
            // аудио: ffmpeg → 16k mono f32 → mel → audio-VAE encode → ref-токены
            let aenc = AudioVaeEncoder::load(&ckpt, dev).map_err(|e| format!("audio encoder: {e}"))?;
            let samples = load_audio_16k(audio_path)?;
            let mel = ltx_log_mel(&[samples], dev).map_err(|e| format!("mel: {e}"))?;
            let audio_ref = aenc.encode(&mel).map_err(|e| format!("audio encode: {e}"))?;
            eprintln!("  lipdub: ref={} audio={} ({} ток)", ref_path.display(), audio_path.display(), audio_ref.dims()[1]);
            let (latent, a_tok) = {
                // lipdub-LoRA мерджится на обе стадии (официально: один stage-объект)
                let lora_view;
                let dit_ckpt: &LtxCheckpoint = if let Some(lp) = &args.lora {
                    let lw = LoraWeights::open(lp, dev, args.lora_strength)
                        .map_err(|e| format!("LoRA {e}"))?;
                    lora_view = ckpt.view_on(dev).with_lora(std::sync::Arc::new(lw));
                    &lora_view
                } else {
                    eprintln!("  [внимание] lipdub без --lora <lipdub-ic-lora> даст слабый sync");
                    &ckpt
                };
                let dit = AvDit::load_with(dit_ckpt, dev, compute, quant_dit, offload)
                    .map_err(|e| format!("AvDit: {e}"))?;
                generate_lipdub_latents(
                    &dit, &up, &v_enc, a_enc_t, &ref1, &ref1_pos, &ref2, &ref2_pos, &audio_ref,
                    fp, hp1, wp1, args.fps, dev,
                )?
            }; // dit дропнут
            let rgb = vae.decode(&latent).map_err(|e| format!("VAE decode: {e}"))?;
            // в mp4 кладём ОРИГИНАЛЬНЫЙ wav (48k stereo) — чище сгенерированной
            // реконструкции; сгенерированное аудио (a_tok) служило кондишеном.
            let _ = &a_tok;
            let wave = load_audio_48k_stereo(audio_path)?;
            return Ok((rgb, Some(wave)));
        }
        if let Some(ref_path) = &args.ref_video {
            // IC-LoRA video→video: reference control-видео (target/downscale) → append.
            // IC-LoRA адаптер мерджится через --lora (в ckpt, см. выше).
            if args.lora.is_none() {
                eprintln!("  [внимание] IC-LoRA обычно требует --lora <ic-lora-адаптер>");
            }
            let (rph, rpw) = (hp * 32 / args.ref_downscale, wp * 32 / args.ref_downscale);
            let mut ref_frames = load_video_frames(ref_path, rph, rpw, out_frames, dev)?;
            if args.ref_preprocess == "canny" {
                ref_frames = apply_canny_frames(&ref_frames, args.canny_low, args.canny_high)?;
                eprintln!("  canny: пороги {}/{} (превью /tmp/synaptix_canny_f0.png)", args.canny_low, args.canny_high);
            } else if args.ref_preprocess == "depth" {
                ref_frames = apply_depth_frames(&ref_frames, &args.depth_model, dev)?;
                eprintln!("  depth: Depth Anything V2 (превью /tmp/synaptix_depth_f0.png)");
            }
            let encoder = VaeEncoder::load(&ckpt, dev).map_err(|e| format!("VAE encoder: {e}"))?;
            let dit = VideoDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                .map_err(|e| format!("VideoDit: {e}"))?;
            eprintln!("  IC-LoRA ref: {} (downscale {} strength {})",
                ref_path.display(), args.ref_downscale, args.ref_strength);
            let rgb = generate_ic_lora_video(
                &dit, &encoder, &vae, &v_enc, &ref_frames, args.ref_downscale, args.ref_strength,
                fp, hp, wp, args.fps, dev,
            )?;
            Ok((rgb, None))
        } else if let Some(vid_path) = &args.video {
            // retake: исходное видео → encode → перегенерация региона [start,end] → decode.
            let (ph, pw) = (hp * 32, wp * 32);
            let frames_t = load_video_frames(vid_path, ph, pw, out_frames, dev)?;
            let encoder = VaeEncoder::load(&ckpt, dev).map_err(|e| format!("VAE encoder: {e}"))?;
            let dit = VideoDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                .map_err(|e| format!("VideoDit: {e}"))?;
            eprintln!("  retake: {} регион [{:.2},{:.2}]с", vid_path.display(), args.retake_start, args.retake_end);
            let rgb = generate_retake_video(
                &dit, &encoder, &vae, &v_enc, &frames_t, args.retake_start, args.retake_end,
                fp, hp, wp, args.fps, dev,
            )?;
            Ok((rgb, None))
        } else if let Some(img_path) = &args.image {
            // image→video: фото → кадр 0 (conditioned). two_stage → HQ (stage1→up→stage2),
            // иначе одна стадия. Сетка stage1 = hp1 (при two_stage = hp/2, выход grid×64).
            let (hp1, wp1) = if two_stage { stage1_grid(hp, wp) } else { (hp, wp) };
            let (ph, pw) = (hp1 * 32, wp1 * 32);
            let raw = synaptix_io::image::load_image(img_path, dev).map_err(|e| format!("image {e}"))?;
            let resized = synaptix_io::image::resize_bilinear(&raw, ph, pw).map_err(|e| format!("resize {e}"))?;
            let img = resized.contiguous()?.affine(2.0, -1.0)?.contiguous()?.reshape(vec![1, 3, 1, ph, pw])?;
            let encoder = VaeEncoder::load(&ckpt, dev).map_err(|e| format!("VAE encoder: {e}"))?;
            let dit = VideoDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                .map_err(|e| format!("VideoDit: {e}"))?;
            eprintln!("  image→video{}: {} (strength {})",
                if two_stage { " HQ two-stage" } else { "" }, img_path.display(), args.image_strength);
            let rgb = if two_stage {
                let up_path = args.upscaler.as_ref()
                    .ok_or("two-stage image→video требует --upscaler")?;
                let ml = SafetensorsLoader::open(&args.model).map_err(|e| format!("stats {e}"))?.with_device(dev);
                let mean = ml.load("vae.per_channel_statistics.mean-of-means").map_err(|e| format!("vae mean: {e}"))?;
                let std = ml.load("vae.per_channel_statistics.std-of-means").map_err(|e| format!("vae std: {e}"))?;
                let up = Upsampler::load(up_path, &mean, &std, dev).map_err(|e| format!("upscaler: {e}"))?;
                generate_image_to_video_two_stage(
                    &dit, &encoder, &up, &vae, &v_enc, &img, fp, hp1, wp1, args.image_strength, args.fps, dev,
                )?
            } else if args.image_frame > 0 {
                // keyframe (append) на пиксель-кадре image_frame
                eprintln!("  keyframe @ pixel-frame {}", args.image_frame);
                generate_keyframe_to_video(
                    &dit, &encoder, &vae, &v_enc, &img, args.image_frame, fp, hp1, wp1, args.image_strength, args.fps, dev,
                )?
            } else {
                generate_image_to_video(
                    &dit, &encoder, &vae, &v_enc, &img, fp, hp1, wp1, args.image_strength, args.fps, dev,
                )?
            };
            Ok((rgb, None))
        } else if two_stage {
            // stage1 (полразрешения) → spatial-upscaler ×2 → stage2-refine → VAE decode.
            // hp/wp — это уже целевая сетка stage2; stage1 идёт на половине.
            let up_path = args.upscaler.as_ref()
                .ok_or("two-stage требует --upscaler <spatial-upscaler.safetensors>")?;
            let ml = SafetensorsLoader::open(&args.model)
                .map_err(|e| format!("stats {e}"))?.with_device(dev);
            let mean = ml.load("vae.per_channel_statistics.mean-of-means").map_err(|e| format!("vae mean: {e}"))?;
            let std = ml.load("vae.per_channel_statistics.std-of-means").map_err(|e| format!("vae std: {e}"))?;
            let mut up = Some(Upsampler::load(up_path, &mean, &std, dev).map_err(|e| format!("upscaler: {e}"))?);
            // сетка stage1 = половина целевой по пространству (упскейлер ×2 восстановит).
            let (hp1, wp1) = stage1_grid(hp, wp);
            // ВАЖНО: DiT'ы строятся ПОСТАДИЙНО во вложенных scope и ДРОПАЮТСЯ —
            // (а) per-stage LoRA (офиц. HQ 0.25/0.5; distilled — без LoRA вовсе),
            // (б) два резидентных mxfp8 (2×20GB) не влезают одновременно,
            // (в) VRAM свободен для тяжёлого HD VAE-decode.
            let ctx_t = v_enc.to_device(dev)?.to_dtype(compute)?;
            let s1 = args.lora_strength_stage1;
            let build_dit = |strength: f32| -> Result<VideoDit, Box<dyn std::error::Error>> {
                if let (Some(lp), true) = (&args.lora, strength > 0.0) {
                    let lw = LoraWeights::open(lp, dev, strength)
                        .map_err(|e| format!("LoRA {e}"))?;
                    let ck = ckpt.view_on(dev).with_lora(std::sync::Arc::new(lw));
                    Ok(VideoDit::load_with(&ck, dev, compute, quant_dit, offload)
                        .map_err(|e| format!("VideoDit(+LoRA {strength}): {e}"))?)
                } else {
                    Ok(VideoDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                        .map_err(|e| format!("VideoDit: {e}"))?)
                }
            };
            // A/V two-stage (официальный distilled): обе стадии денойзят видео+аудио
            // совместно (AvDit); stage2 ре-нойзит видео (upscaled) И аудио-латент.
            if let (true, Some(a_enc_t)) = (audio && !guided, &a_enc) {
                let (e1, e2) = if args.lora.is_some() { (s1, lora_s2) } else { (0.0, 0.0) };
                let build_avdit = |strength: f32| -> Result<AvDit, Box<dyn std::error::Error>> {
                    if let (Some(lp), true) = (&args.lora, strength > 0.0) {
                        let lw = LoraWeights::open(lp, dev, strength).map_err(|e| format!("LoRA {e}"))?;
                        let ck = ckpt.view_on(dev).with_lora(std::sync::Arc::new(lw));
                        Ok(AvDit::load_with(&ck, dev, compute, quant_dit, offload)
                            .map_err(|e| format!("AvDit(+LoRA {strength}): {e}"))?)
                    } else {
                        Ok(AvDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                            .map_err(|e| format!("AvDit: {e}"))?)
                    }
                };
                let (latent, a_tok) = if e1 == e2 {
                    let dit = build_avdit(e1)?;
                    let ts1 = std::time::Instant::now();
                    let (l1, a1) = denoise_av(&dit, &v_enc, a_enc_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, None, args.fps, dev, v_nag, &[], None, &DenoiseHooks::none())?;
                    let _ = synaptix_core::device::cuda::synchronize(0);
                    eprintln!("  stage1-denoise: {:.1}s", ts1.elapsed().as_secs_f32());
                    if refine {
                        let tu = std::time::Instant::now();
                        let l2 = up.as_ref().expect("upscaler").upsample(&l1)?.to_dtype(compute)?;
                        up = None; // upscaler (~1GB VRAM) не нужен на stage2-refine
                        synaptix_core::tensor::ops::conv_filter_cache_clear(); // krsc-копии upscaler → VRAM под stage2
                        let _ = synaptix_core::device::cuda::synchronize(0);
                        eprintln!("  upsample: {:.1}s", tu.elapsed().as_secs_f32());
                        let ts2 = std::time::Instant::now();
                        let r = denoise_av(&dit, &v_enc, a_enc_t, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS, Some(&l2), Some(&a1), args.fps, dev, None, &[], None, &DenoiseHooks::none())?;
                        let _ = synaptix_core::device::cuda::synchronize(0);
                        eprintln!("  stage2-denoise: {:.1}s", ts2.elapsed().as_secs_f32());
                        r
                    } else {
                        (up.as_ref().expect("upscaler").upsample(&l1)?, a1)
                    }
                } else {
                    let (l1, a1) = {
                        let dit1 = build_avdit(e1)?;
                        denoise_av(&dit1, &v_enc, a_enc_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, None, args.fps, dev, v_nag, &[], None, &DenoiseHooks::none())?
                    };
                    if refine {
                        let l2 = up.as_ref().expect("upscaler").upsample(&l1)?.to_dtype(compute)?;
                        up = None; // upscaler (~1GB VRAM) не нужен на stage2-refine
                        synaptix_core::tensor::ops::conv_filter_cache_clear(); // krsc-копии upscaler → VRAM под stage2
                        let dit2 = build_avdit(e2)?;
                        denoise_av(&dit2, &v_enc, a_enc_t, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS, Some(&l2), Some(&a1), args.fps, dev, None, &[], None, &DenoiseHooks::none())?
                    } else {
                        (up.as_ref().expect("upscaler").upsample(&l1)?, a1)
                    }
                }; // dit дропнут → VRAM свободен под VAE/декод
                drop(up);
                synaptix_core::tensor::ops::conv_filter_cache_gc();
                // дефраг: VAE-веса, созданные ДО denoise, размазаны по сегментам
                // пула и не дают trim'у их вернуть (live 0.8GB удерживал
                // reserved ~9GB → budget VAE падал втрое: 20s vae 34→95s).
                // Пересоздание после полного трима кладёт веса компактно.
                drop(vae);
                if let Device::Cuda(o) = dev {
                    let _ = synaptix_core::device::cuda::synchronize_all(o);
                    let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(o);
                }
                let vae = VaeDecoder::load(&ckpt, dev).map_err(|e| format!("VAE: {e}"))?;
                let tv = std::time::Instant::now();
                let rgb = vae.decode(&latent).map_err(|e| format!("VAE decode: {e}"))?;
                let _ = synaptix_core::device::cuda::synchronize(0);
                eprintln!("  vae-decode: {:.1}s", tv.elapsed().as_secs_f32());
                let ta = std::time::Instant::now();
                let audio_vae = AudioVaeDecoder::load(&ckpt, dev).map_err(|e| format!("audio VAE: {e}"))?;
                let vocoder = VocoderWithBwe::load(&args.model, dev).map_err(|e| format!("vocoder: {e}"))?;
                let wave = decode_audio_tokens(&audio_vae, &vocoder, &a_tok)?;
                let _ = synaptix_core::device::cuda::synchronize(0);
                eprintln!("  audio-decode+vocoder: {:.1}s", ta.elapsed().as_secs_f32());
                return Ok((rgb, Some(wave)));
            }
            let latent = if guided {
                // guided: stage1 = dev БЕЗ LoRA (CFG/STG), stage2 = +distilled-LoRA.
                let neg = neg_enc.as_ref().expect("neg-context при guided");
                let l1 = {
                    let dit1 = build_dit(0.0)?;
                    let neg_t = neg.to_device(dev)?.to_dtype(compute)?;
                    let sg = synaptix_video_ltx23::pipeline::ltx2_sigmas(args.steps, fp * hp1 * wp1);
                    synaptix_video_ltx23::pipeline::denoise_video_guided(
                        &dit1, &ctx_t, &neg_t, &guider, fp, hp1, wp1, &sg, None, args.fps, dev,
                        None, &DenoiseHooks::none(),
                    )?
                }; // dit1 дропнут
                let l2 = up.as_ref().expect("upscaler").upsample(&l1)?.to_dtype(compute)?;
                        up = None; // upscaler (~1GB VRAM) не нужен на stage2-refine
                        synaptix_core::tensor::ops::conv_filter_cache_clear(); // krsc-копии upscaler → VRAM под stage2
                let dit2 = build_dit(if args.lora.is_some() { lora_s2 } else { 0.0 })?;
                denoise(&dit2, &ctx_t, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS, Some(&l2), args.fps, dev, None, &DenoiseHooks::none())?
            } else {
                // distilled two-stage: офиц. БЕЗ LoRA (s1=0,s2=0 если не задано иное).
                let (e1, e2) = if args.lora.is_some() { (s1, lora_s2) } else { (0.0, 0.0) };
                if e1 == e2 {
                    // один dit на обе стадии (без двойной загрузки модели)
                    let dit = build_dit(e1)?;
                    let l1 = denoise(&dit, &ctx_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, args.fps, dev, None, &DenoiseHooks::none())?;
                    if refine {
                        let l2 = up.as_ref().expect("upscaler").upsample(&l1)?.to_dtype(compute)?;
                        up = None; // upscaler (~1GB VRAM) не нужен на stage2-refine
                        synaptix_core::tensor::ops::conv_filter_cache_clear(); // krsc-копии upscaler → VRAM под stage2
                        denoise(&dit, &ctx_t, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS, Some(&l2), args.fps, dev, None, &DenoiseHooks::none())?
                    } else {
                        up.as_ref().expect("upscaler").upsample(&l1)?
                    }
                } else {
                    // разные strengths → постадийные dit (HQ-паттерн 0.25/0.5)
                    let l1 = {
                        let dit1 = build_dit(e1)?;
                        denoise(&dit1, &ctx_t, fp, hp1, wp1, &DISTILLED_SIGMAS, None, args.fps, dev, None, &DenoiseHooks::none())?
                    }; // dit1 дропнут
                    if refine {
                        let l2 = up.as_ref().expect("upscaler").upsample(&l1)?.to_dtype(compute)?;
                        up = None; // upscaler (~1GB VRAM) не нужен на stage2-refine
                        synaptix_core::tensor::ops::conv_filter_cache_clear(); // krsc-копии upscaler → VRAM под stage2
                        let dit2 = build_dit(e2)?;
                        denoise(&dit2, &ctx_t, fp, hp1 * 2, wp1 * 2, &STAGE2_SIGMAS, Some(&l2), args.fps, dev, None, &DenoiseHooks::none())?
                    } else {
                        up.as_ref().expect("upscaler").upsample(&l1)?
                    }
                }
            }; // dit'ы дропнуты → VRAM свободен под VAE
            drop(up);
            synaptix_core::tensor::ops::conv_filter_cache_gc();
            // дефраг пула перед VAE — см. avdit-ветку выше
            drop(vae);
            if let Device::Cuda(o) = dev {
                let _ = synaptix_core::device::cuda::synchronize_all(o);
                let _ = synaptix_core::memory::cuda_pool::hard_trim_cuda_mempool_device(o);
            }
            let vae = VaeDecoder::load(&ckpt, dev).map_err(|e| format!("VAE: {e}"))?;
            let rgb = vae.decode(&latent).map_err(|e| format!("VAE decode: {e}"))?;
            Ok((rgb, None))
        } else if let Some(a_enc) = a_enc {
            let dit = AvDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                .map_err(|e| format!("AvDit: {e}"))?;
            let audio_vae = AudioVaeDecoder::load(&ckpt, dev).map_err(|e| format!("audio VAE: {e}"))?;
            let vocoder = VocoderWithBwe::load(&args.model, dev).map_err(|e| format!("vocoder: {e}"))?;
            let (rgb, wave) =
                generate_av(&dit, &vae, &audio_vae, &vocoder, &v_enc, &a_enc, fp, hp, wp, args.fps, dev)?;
            Ok((rgb, Some(wave)))
        } else {
            let dit = VideoDit::load_with(&ckpt, dev, compute, quant_dit, offload)
                .map_err(|e| format!("VideoDit: {e}"))?;
            let rgb = generate_video(&dit, &vae, &v_enc, fp, hp, wp, args.fps, dev)?;
            Ok((rgb, None))
        }
    })?;
    let _ = synaptix_core::device::cuda::synchronize(0);
    eprintln!(
        "  генерация: {:.1}s → {} кадров {}×{}",
        t_gen.elapsed().as_secs_f32(), rgb.dims()[2], rgb.dims()[3], rgb.dims()[4]
    );

    // ── 3. кадры → PPM, [wave → wav], ffmpeg-мукс → mp4 ──
    write_mp4(&rgb, wave.as_ref(), args.fps, &args.output)?;
    Ok(())
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
/// формы (ближе=белее, [−1,1]). Превью кадра 0 → /tmp/synaptix_depth_f0.png.
fn apply_depth_frames(
    frames: &synaptix_core::tensor::Tensor,
    model_dir: &std::path::Path,
    dev: Device,
) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let m = synaptix_depth_anything::DepthAnything::load(model_dir, dev)
        .map_err(|e| format!("depth model: {e}"))?;
    let (f, h, w) = (frames.dims()[2], frames.dims()[3], frames.dims()[4]);
    let mut out: Vec<synaptix_core::tensor::Tensor> = Vec::with_capacity(f);
    for fi in 0..f {
        let fr = frames.narrow(2, fi, 1)?.contiguous()?.reshape(vec![3, h, w])?
            .affine(0.5, 0.5)?; // [−1,1] → [0,1]
        let d = m.depth_rgb(&fr).map_err(|e| format!("depth: {e}"))?;
        if fi == 0 {
            let _ = synaptix_io::image::save_image(&d, "/tmp/synaptix_depth_f0.png");
        }
        out.push(d.affine(2.0, -1.0)?.reshape(vec![1, 3, 1, h, w])?);
    }
    let refs: Vec<&synaptix_core::tensor::Tensor> = out.iter().collect();
    Ok(synaptix_core::tensor::Tensor::cat(&refs, 2)?.contiguous()?)
}

/// Canny по всем кадрам `[1,3,F,H,W]` ([−1,1]) → контурный control-сигнал той же
/// формы (белые рёбра на чёрном, [−1,1]). Превью кадра 0 → /tmp/synaptix_canny_f0.png.
fn apply_canny_frames(
    frames: &synaptix_core::tensor::Tensor,
    low: f32,
    high: f32,
) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let (f, h, w) = (frames.dims()[2], frames.dims()[3], frames.dims()[4]);
    let mut out: Vec<synaptix_core::tensor::Tensor> = Vec::with_capacity(f);
    for fi in 0..f {
        let fr = frames.narrow(2, fi, 1)?.contiguous()?.reshape(vec![3, h, w])?
            .affine(0.5, 0.5)?; // [−1,1] → [0,1]
        let edges = synaptix_io::image::canny_rgb(&fr, low, high).map_err(|e| format!("canny: {e}"))?;
        if fi == 0 {
            let _ = synaptix_io::image::save_image(&edges, "/tmp/synaptix_canny_f0.png");
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

/// Загрузить кадры видео `path` → `[1,3,F,ph,pw]` ([−1,1]). ffmpeg извлекает все
/// кадры (scale ph×pw) в temp-PNG, грузим первые `n`, стекаем по времени.
fn load_video_frames(
    path: &std::path::Path,
    ph: usize,
    pw: usize,
    n: usize,
    dev: Device,
) -> Result<synaptix_core::tensor::Tensor, Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("synaptix_retake_in");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let status = Command::new("ffmpeg")
        .args(["-y", "-i"]).arg(path)
        .args(["-vf", &format!("scale={pw}:{ph}")])
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
        return Err("retake: видео не дало кадров".into());
    }
    let refs: Vec<&synaptix_core::tensor::Tensor> = frames.iter().collect();
    Ok(synaptix_core::tensor::Tensor::cat(&refs, 2)?.contiguous()?)
}

/// RGB `[1,3,F,H,W]` (+опц. стерео wave `[1,2,L]`) → mp4 через ffmpeg (PPM-кадры
/// + WAV в temp-директории).
fn write_mp4(
    rgb: &synaptix_core::tensor::Tensor,
    wave: Option<&synaptix_core::tensor::Tensor>,
    fps: f64,
    out: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join("synaptix_video_frames");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;
    let frames = rgb_to_frames(rgb)?;
    for (i, fr) in frames.iter().enumerate() {
        let (h, w) = (fr.dims()[1], fr.dims()[2]);
        let planar: Vec<f32> = fr.reshape(vec![3 * h * w])?.to_vec1::<f32>()?;
        let mut buf = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                for c in 0..3 {
                    buf.push((planar[c * h * w + y * w + x].clamp(0.0, 1.0) * 255.0) as u8);
                }
            }
        }
        std::fs::write(dir.join(format!("f{i:05}.ppm")), buf)?;
    }
    let frame_glob = dir.join("f%05d.ppm");
    let fr_s = format!("{fps}");
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-framerate", &fr_s, "-i"]).arg(&frame_glob);
    let wav_path = dir.join("audio.wav");
    if let Some(wave) = wave {
        write_wav(&wav_path, wave, 48000)?;
        cmd.arg("-i").arg(&wav_path).args(["-c:a", "aac", "-shortest"]);
    }
    cmd.args(["-c:v", "libx264", "-pix_fmt", "yuv420p"]).arg(out);
    let status = cmd.status()?;
    if status.success() {
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

use std::path::PathBuf;

use synaptix_audio::io::{read_wav_stereo_f32, write_wav_interleaved_f32, write_wav_mono_f32};
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_music_acestep::ar::CodesGenOptions;
use synaptix_music_acestep::dcw::{DcwCorrector, DcwMode};
use synaptix_music_acestep::pipeline::{
    apply_norm, generate_music, EditMode, EditOptions, GenExtras, MusicPaths, NormMode, SamplerOptions,
};
use synaptix_music_acestep::DitVariant;
use synaptix_music_acestep::text_encoder::TRACK_NAMES;
use synaptix_music_acestep::vae::AceStepVae;

use crate::commands::device;

pub struct MusicArgs {
    pub caption: String,
    pub lyrics: String,
    pub output: PathBuf,
    pub models: PathBuf,
    pub lm: Option<PathBuf>,
    pub text_encoder: Option<PathBuf>,
    pub dit: Option<PathBuf>,
    pub vae: Option<PathBuf>,
    pub duration: String,
    pub steps: Option<usize>,
    pub cfg: Option<f32>,
    pub shift: Option<f32>,
    pub seed: u64,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub lm_cfg: f32,
    pub use_cot: bool,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub quant: Option<String>,
    pub quant_encoder: Option<String>,
    pub retake_variance: f32,
    pub retake_seed: u64,
    pub mode: String,
    pub track: String,
    pub src_audio: Option<PathBuf>,
    pub repaint_start: f32,
    pub repaint_end: f32,
    pub repaint_strength: f32,
    pub edit_n_min: f32,
    pub edit_n_max: f32,
    pub edit_n_avg: usize,
    pub edit_source_caption: String,
    pub edit_source_lyric: String,
    pub use_ar: bool,
    pub bpm: Option<u32>,
    pub keyscale: String,
    pub timesig: String,
    pub norm: String,
    /// >1 — прогоны подряд с резидентным кэшем компонентов.
    pub repeat: u32,
    pub dcw: bool,
    pub dcw_mode: String,
    pub dcw_preset: String,
    pub dcw_scaler: Option<f32>,
    pub dcw_high_scaler: Option<f32>,
    pub channels: u16,
    pub save_latent: Option<PathBuf>,
}

fn random_seed() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64 ^ std::process::id() as u64)
        .unwrap_or(1)
        | 1
}

fn parse_dcw(args: &MusicArgs) -> Result<DcwCorrector, String> {
    let mode = match args.dcw_mode.as_str() {
        "low" => DcwMode::Low,
        "high" => DcwMode::High,
        "double" => DcwMode::Double,
        "pix" => DcwMode::Pix,
        o => return Err(format!("--dcw-mode: low | high | double | pix, а не `{o}`")),
    };
    let (scaler, high_scaler) = match args.dcw_preset.as_str() {
        "think" => (0.02, 0.06),
        "no-think" | "nothink" => (0.05, 0.02),
        o => return Err(format!("--dcw-preset: think | no-think, а не `{o}`")),
    };
    Ok(DcwCorrector {
        enabled: args.dcw,
        mode,
        scaler: args.dcw_scaler.unwrap_or(scaler),
        high_scaler: args.dcw_high_scaler.unwrap_or(high_scaler),
    })
}

pub fn run(args: MusicArgs) -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let pick = |o: Option<PathBuf>, name: &str| o.unwrap_or_else(|| args.models.join(name));
    let lm = pick(args.lm.clone(), "acestep_5hz_lm_1.7b.syn");
    let text_encoder = pick(args.text_encoder.clone(), "qwen3-embedding-0.6b.syn");
    let dit = pick(args.dit.clone(), "acestep_v15_xl_base.syn");
    let vae = pick(args.vae.clone(), "acestep_vae.syn");
    for (label, p) in [("lm", &lm), ("text-encoder", &text_encoder), ("dit", &dit), ("vae", &vae)] {
        if !p.exists() {
            return Err(format!("{label} bundle not found: {} (use --models <dir> or --{label} <path>)", p.display()).into());
        }
    }
    let paths = MusicPaths { lm: &lm, text_encoder: &text_encoder, dit: &dit, vae: &vae };

    let auto_duration = args.duration.trim().eq_ignore_ascii_case("auto");
    let mut duration_sec: u32 = if auto_duration {
        0 // 0 → Phase-1 CoT сам предсказывает длительность
    } else {
        args.duration.trim().parse()
            .map_err(|_| format!("--duration: ожидалось 'auto' или число секунд, получено '{}'", args.duration))?
    };
    if args.channels != 1 && args.channels != 2 {
        return Err(format!("--channels: 1 | 2, а не {}", args.channels).into());
    }
    let variant = DitVariant::detect(&dit);
    let steps = args.steps.unwrap_or(variant.default_steps());
    let cfg = args.cfg.unwrap_or(variant.default_cfg());
    let shift = args.shift.unwrap_or(variant.default_shift());
    let dcw = parse_dcw(&args)?;
    let seed = if args.seed == 0 { random_seed() } else { args.seed };
    let retake_seed = if args.retake_seed == 0 { random_seed() } else { args.retake_seed };

    let dev = device::resolve(&args.device);
    // DiT-рендеринг dtype (AR-LM всегда F32 для точных кодов). Дефолт bf16 —
    // tensor-core GEMM'ы DiT ~2-4× быстрее F32, качество подтверждено; f32 = макс.точность.
    let compute = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") | None => DType::BF16,
        Some("f32") => DType::F32,
        Some(o) => return Err(format!("unknown compute-dtype {o}").into()),
    };
    // --quant (веса DiT) / --quant-encoder (веса LM + text-enc). none/None → compute
    // (dense, бит-в-бит как раньше). nvfp4/mxfp8 → квант (pipeline сам форсит F16-compute
    // квантуемому энкодеру). DiT-квант идёт на bf16-compute (как LTX).
    let parse_q = |o: Option<&str>, what: &str| -> Result<DType, Box<dyn std::error::Error>> {
        match o {
            None | Some("none") => Ok(compute),
            Some(s) => synaptix_core::precision::parse_dtype(s)
                .ok_or_else(|| format!("--{what}: ожидалось none|nvfp4|mxfp8|f16|bf16|f32, получено '{s}'").into()),
        }
    };
    let dit_quant = parse_q(args.quant.as_deref(), "quant")?;
    let enc_quant = parse_q(args.quant_encoder.as_deref(), "quant-encoder")?;

    let opts = SamplerOptions { steps, shift, guidance_scale: cfg, dcw };
    let copts = CodesGenOptions {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_p: args.min_p,
        cfg_scale: args.lm_cfg,
        seed,
        ..CodesGenOptions::default()
    };

    eprintln!(
        "synaptix music: \"{}\" lyrics={}b dur={} dit={variant:?} steps={steps} cfg={cfg} shift={shift} lm_cfg={} seed={seed} dcw={} ({dev:?}, compute={compute:?}, dit_quant={dit_quant:?}, enc_quant={enc_quant:?})",
        args.caption, args.lyrics.len(), args.duration, args.lm_cfg, opts.dcw.is_active()
    );
    let mode = match args.mode.as_str() {
        "retake" => EditMode::Retake,
        "repaint" | "extend" => EditMode::Repaint,
        "edit" => EditMode::Edit,
        "extract" => EditMode::Extract,
        "cover" => EditMode::Cover,
        _ if args.retake_variance > 0.0 => EditMode::Retake,
        _ => EditMode::Text2Music,
    };
    if mode == EditMode::Extract && !TRACK_NAMES.contains(&args.track.as_str()) {
        return Err(format!("--track: ожидалось одно из {TRACK_NAMES:?}, получено '{}'", args.track).into());
    }
    // src_latent для repaint/extend/edit/extract/cover: исходное аудио → VAE → [1,T,64].
    let src_latent = if matches!(
        mode,
        EditMode::Repaint | EditMode::Edit | EditMode::Extract | EditMode::Cover
    ) {
        let sp = args.src_audio.as_ref().ok_or("режим требует --src-audio <wav>")?;
        // Стерео как есть (панорама помогает extract), в [-1, 1] — как
        // `_normalize_audio_to_stereo_48k` у ACE-Step.
        let (mut flat, asr) = read_wav_stereo_f32(sp)?;
        if asr != 48000 {
            return Err(format!("--src-audio: ожидался 48 kHz, получено {asr} Hz").into());
        }
        flat.iter_mut().for_each(|s| *s = s.clamp(-1.0, 1.0));
        let n = flat.len() / 2;
        let at = Tensor::from_vec(flat, vec![1usize, 2, n], dev)?.to_dtype(compute)?;
        let vae_enc = AceStepVae::open(&vae, dev)?;
        let lat = vae_enc.encode_mean(&at)?; // [1,64,T]
        if auto_duration && args.mode != "extend" {
            duration_sec = ((lat.dims()[2] as f32 / 25.0).round() as u32).max(1);
            eprintln!("synaptix music: длительность по исходнику — {duration_sec} с");
        }
        Some(lat.transpose(1, 2)?.contiguous()?) // [1,T,64]
    } else {
        None
    };
    let edit = EditOptions {
        mode,
        track_name: args.track.clone(),
        retake_variance: args.retake_variance,
        retake_seed,
        src_latent,
        repaint_start_sec: args.repaint_start,
        repaint_end_sec: args.repaint_end,
        repaint_strength: args.repaint_strength,
        edit_n_min: args.edit_n_min,
        edit_n_max: args.edit_n_max,
        edit_n_avg: args.edit_n_avg.max(1),
        edit_source_caption: args.edit_source_caption.clone(),
        edit_source_lyric: args.edit_source_lyric.clone(),
    };
    let norm_mode = match args.norm.as_str() {
        "off" => NormMode::Off,
        "rms" => NormMode::Rms,
        _ => NormMode::Peak,
    };
    let s = |x: &str| if x.trim().is_empty() { None } else { Some(x.trim().to_string()) };
    let extras = GenExtras {
        use_ar: args.use_ar,
        bpm: args.bpm,
        keyscale: s(&args.keyscale),
        timesig: s(&args.timesig),
        norm_mode,
        on_stage: None,
    };
    // --repeat: резидентный кэш LM/TE/DiT/VAE переживает прогоны — так
    // работают ноды synthos с «Держать в памяти».
    let mut cache = (args.repeat > 1).then(synaptix_music_acestep::pipeline::MusicComponentCache::default);
    let mut result = None;
    for round in 0..args.repeat.max(1) {
        let copts = CodesGenOptions { seed: copts.seed + round as u64, ..copts.clone() };
        let t0 = std::time::Instant::now();
        let (samples, sr, latent) = generate_music(
            &paths, &args.caption, &args.lyrics, duration_sec, dev, compute, dit_quant, enc_quant, &opts, &copts, args.use_cot, &edit, &extras, cache.as_mut(),
        )?;
        let dur = samples.len() as f32 / sr as f32;
        let infer = t0.elapsed().as_secs_f32();
        eprintln!(
            "synaptix music: прогон {}/{}: {dur:.1}s audio in {infer:.1}s (RTF={:.3})",
            round + 1,
            args.repeat.max(1),
            infer / dur.max(1e-6)
        );
        result = Some((samples, sr, latent));
    }
    let (samples, sr, latent) = result.expect("хотя бы один прогон");
    if let Some(path) = &args.save_latent {
        crate::commands::song::save_latent(path, &latent)?;
        eprintln!("synaptix music: латент {:?} → {}", latent.dims(), path.display());
    }
    if args.channels == 2 {
        let audio = AceStepVae::open(&vae, dev)?.decode_tiled(&latent, 500, 32)?;
        let channel = |c: usize| -> Result<Vec<f32>, Box<dyn std::error::Error>> {
            Ok(audio.narrow(1, c, 1)?.contiguous()?.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?)
        };
        let (left, right) = (channel(0)?, channel(1)?);
        let mut stereo: Vec<f32> = left.iter().zip(&right).flat_map(|(l, r)| [*l, *r]).collect();
        apply_norm(&mut stereo, norm_mode);
        write_wav_interleaved_f32(&args.output, &stereo, sr, 2)?;
    } else {
        write_wav_mono_f32(&args.output, &samples, sr)?;
    }
    eprintln!("synaptix music: wrote {}", args.output.display());
    Ok(())
}

use std::path::PathBuf;

use synaptix_audio::io::{read_wav_mono_f32, write_wav_mono_f32};
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_music_acestep::ar::CodesGenOptions;
use synaptix_music_acestep::pipeline::{generate_music, EditMode, EditOptions, GenExtras, MusicPaths, NormMode, SamplerOptions};
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
    pub steps: usize,
    pub cfg: f32,
    pub shift: f32,
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
    pub src_audio: Option<PathBuf>,
    pub repaint_start: f32,
    pub repaint_end: f32,
    pub repaint_strength: f32,
    pub edit_n_min: f32,
    pub edit_n_max: f32,
    pub edit_source_caption: String,
    pub edit_source_lyric: String,
    pub use_ar: bool,
    pub bpm: Option<u32>,
    pub keyscale: String,
    pub timesig: String,
    pub norm: String,
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

    let duration_sec: u32 = if args.duration.trim().eq_ignore_ascii_case("auto") {
        0 // 0 → Phase-1 CoT сам предсказывает длительность
    } else {
        args.duration.trim().parse()
            .map_err(|_| format!("--duration: ожидалось 'auto' или число секунд, получено '{}'", args.duration))?
    };

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

    let opts = SamplerOptions {
        steps: args.steps,
        shift: args.shift,
        guidance_scale: args.cfg,
        ..SamplerOptions::default()
    };
    let copts = CodesGenOptions {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        min_p: args.min_p,
        cfg_scale: args.lm_cfg,
        seed: args.seed,
        ..CodesGenOptions::default()
    };

    eprintln!(
        "synaptix music: \"{}\" lyrics={}b dur={} steps={} cfg={} shift={} lm_cfg={} ({dev:?}, compute={compute:?}, dit_quant={dit_quant:?}, enc_quant={enc_quant:?})",
        args.caption, args.lyrics.len(), args.duration, args.steps, args.cfg, args.shift, args.lm_cfg
    );
    let mode = match args.mode.as_str() {
        "retake" => EditMode::Retake,
        "repaint" | "extend" => EditMode::Repaint,
        "edit" => EditMode::Edit,
        "extract" | "cover" => EditMode::Extract,
        _ if args.retake_variance > 0.0 => EditMode::Retake,
        _ => EditMode::Text2Music,
    };
    // src_latent для repaint/extend/edit/extract: исходное аудио → VAE → [1,T,64].
    let src_latent = if matches!(mode, EditMode::Repaint | EditMode::Edit | EditMode::Extract) {
        let sp = args.src_audio.as_ref().ok_or("режим требует --src-audio <wav>")?;
        let (mono, asr) = read_wav_mono_f32(sp)?;
        if asr != 48000 {
            return Err(format!("--src-audio: ожидался 48 kHz, получено {asr} Hz").into());
        }
        let n = mono.len();
        let mut flat = Vec::with_capacity(2 * n);
        flat.extend_from_slice(&mono);
        flat.extend_from_slice(&mono);
        let at = Tensor::from_vec(flat, vec![1usize, 2, n], dev)?.to_dtype(compute)?;
        let vae_enc = AceStepVae::open(&vae, dev)?;
        let lat = vae_enc.encode_mean(&at)?; // [1,64,T]
        Some(lat.transpose(1, 2)?.contiguous()?) // [1,T,64]
    } else {
        None
    };
    let edit = EditOptions {
        mode,
        retake_variance: args.retake_variance,
        retake_seed: args.retake_seed,
        src_latent,
        repaint_start_sec: args.repaint_start,
        repaint_end_sec: args.repaint_end,
        repaint_strength: args.repaint_strength,
        edit_n_min: args.edit_n_min,
        edit_n_max: args.edit_n_max,
        edit_n_avg: 1,
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
    };
    let t0 = std::time::Instant::now();
    let (samples, sr, _latent) = generate_music(
        &paths, &args.caption, &args.lyrics, duration_sec, dev, compute, dit_quant, enc_quant, &opts, &copts, args.use_cot, &edit, &extras,
    )?;
    let dur = samples.len() as f32 / sr as f32;
    let infer = t0.elapsed().as_secs_f32();
    eprintln!("synaptix music: {dur:.1}s audio in {infer:.1}s (RTF={:.3})", infer / dur.max(1e-6));
    write_wav_mono_f32(&args.output, &samples, sr)?;
    eprintln!("synaptix music: wrote {}", args.output.display());
    Ok(())
}

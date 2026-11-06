use std::path::{Path, PathBuf};

use synaptix_audio::io::write_wav_mono_f32;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::{OmniVoiceGenerationConfig, OmniVoicePipeline};
use synaptix_tts_voxcpm::{GenerateOptions, VoxCpmPipeline};

use crate::commands::device;

pub struct SpeakArgs {
    pub bundle: PathBuf,
    pub text: String,
    pub output: PathBuf,
    pub reference: Option<PathBuf>,
    pub prompt_wav: Option<PathBuf>,
    pub prompt_text: Option<String>,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub cfg: Option<f32>,
    pub steps: Option<usize>,
    pub seed: u64,
    pub max_len: usize,
    pub engine: String,
    pub min_len: Option<usize>,
    pub retry_ratio: Option<f32>,
    pub streaming_prefix_len: Option<usize>,
    pub language: Option<String>,
    pub instruct: Option<String>,
    pub speed: f64,
    pub t_shift: Option<f32>,
    pub layer_penalty: Option<f32>,
    pub no_denoise: bool,
    pub target_tokens: Option<usize>,
    pub voice: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    VoxCpm,
    OmniVoice,
    Audio8,
    Supertonic,
    Leokani,
}

fn detect_engine(bundle: &Path, requested: &str) -> Result<Engine, String> {
    match requested {
        "voxcpm" => return Ok(Engine::VoxCpm),
        "omnivoice" => return Ok(Engine::OmniVoice),
        "audio8" => return Ok(Engine::Audio8),
        "supertonic" => return Ok(Engine::Supertonic),
        "leokani" => return Ok(Engine::Leokani),
        "auto" => {}
        o => return Err(format!("--engine: auto | voxcpm | omnivoice | audio8 | supertonic | leokani, а не `{o}`")),
    }
    let id = synaptix_bundle::Bundle::open(bundle).map(|b| b.id().to_lowercase()).unwrap_or_default();
    let name = bundle.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let has = |k: &str| id.contains(k) || name.contains(k);
    if has("leokani") || synaptix_bundle::Bundle::open(bundle).map(|b| b.meta().arch == "leokani").unwrap_or(false) {
        return Ok(Engine::Leokani);
    }
    if has("supertonic") || synaptix_bundle::Bundle::open(bundle).map(|b| b.meta().arch == "supertonic").unwrap_or(false) {
        return Ok(Engine::Supertonic);
    }
    if has("audio8") || synaptix::facade::arch::model_type(bundle).as_deref() == Some("arktts") {
        return Ok(Engine::Audio8);
    }
    if has("vibevoice") {
        return Err("VibeVoice — многоголосый синтез: используйте `synaptix podcast`".into());
    }
    Ok(if has("omnivoice") { Engine::OmniVoice } else { Engine::VoxCpm })
}

pub fn run(args: SpeakArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.bundle.exists() {
        return Err(format!("bundle not found: {}", args.bundle.display()).into());
    }
    let engine = detect_engine(&args.bundle, &args.engine)?;
    let dev = device::resolve(&args.device);
    let dtype = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(other) => return Err(format!("unknown compute-dtype {other}").into()),
        None => match (dev, engine) {
            (Device::Cpu, _) => DType::F32,
            (_, Engine::OmniVoice) => DType::F16,
            _ => DType::BF16,
        },
    };
    let t0 = std::time::Instant::now();
    eprintln!("synaptix speak: {} [{engine:?}] (compute={dtype:?}, {dev:?})", args.bundle.display());
    let (pcm, rate) = match engine {
        Engine::VoxCpm => speak_voxcpm(&args, dev, dtype, t0)?,
        Engine::OmniVoice => speak_omnivoice(&args, dev, dtype, t0)?,
        Engine::Audio8 => speak_audio8(&args, dev, dtype, t0)?,
        Engine::Supertonic => speak_supertonic(&args, dev, t0)?,
        Engine::Leokani => speak_leokani(&args, dev, t0)?,
    };
    write_wav_mono_f32(&args.output, &pcm, rate)?;
    eprintln!("synaptix speak: wrote {}", args.output.display());
    Ok(())
}

fn report(pcm: &[f32], rate: u32, t: std::time::Instant) {
    let infer = t.elapsed().as_secs_f32();
    let dur = pcm.len() as f32 / rate as f32;
    eprintln!("synaptix speak: {:.2}s audio in {:.2}s (RTF={:.3})", dur, infer, infer / dur.max(1e-6));
}

/// Leokani-tts 7M (английский): `--voice` dala | pono, `--speed` — темп.
fn speak_leokani(args: &SpeakArgs, dev: Device, t0: std::time::Instant) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let pipe = synaptix_tts_leokani::LeokaniPipeline::from_syn(&args.bundle, dev)?;
    eprintln!("synaptix speak: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let t1 = std::time::Instant::now();
    let pcm = pipe.synthesize(&args.text, args.voice.as_deref().unwrap_or("dala"), args.speed as f32)?;
    report(&pcm, pipe.sample_rate(), t1);
    Ok((pcm, pipe.sample_rate()))
}

/// Supertonic 3: `--voice` — готовый голос (F1…F5, M1…M5), `--language` —
/// код языка (по умолчанию по письму: ru/en), `--steps` (8), `--speed` — темп.
fn speak_supertonic(args: &SpeakArgs, dev: Device, t0: std::time::Instant) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    use synaptix_tts_supertonic::{Options, SupertonicPipeline};
    let pipe = SupertonicPipeline::from_syn(&args.bundle, dev)?;
    eprintln!("synaptix speak: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let name = args.voice.clone().unwrap_or_else(|| "M1".into());
    let voice = pipe.voice(&name).ok_or_else(|| format!("голос {name}: есть {}", pipe.voice_names().join(", ")))?.clone();
    let lang = args.language.clone().unwrap_or_else(|| {
        let cyr = args.text.chars().filter(|c| matches!(c, 'а'..='я' | 'А'..='Я' | 'ё' | 'Ё')).count();
        if cyr * 2 > args.text.chars().filter(|c| c.is_alphabetic()).count() { "ru".into() } else { "en".into() }
    });
    let opts = Options { steps: args.steps.unwrap_or(8), speed: 1.05 * args.speed as f32, seed: args.seed, ..Options::default() };
    let t1 = std::time::Instant::now();
    let pcm = pipe.synthesize(&args.text, &lang, &voice, &opts)?;
    report(&pcm, pipe.sample_rate(), t1);
    Ok((pcm, pipe.sample_rate()))
}

/// Audio8-TTS: `--reference` — образец голоса, `--prompt-text` — его точная
/// расшифровка (без неё клон невозможен); `--speed` — растяжение времени.
fn speak_audio8(
    args: &SpeakArgs,
    dev: Device,
    dtype: DType,
    t0: std::time::Instant,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    use synaptix_tts_audio8::{Audio8Pipeline, GenerateOptions as A8Options};
    let pipe = Audio8Pipeline::from_syn_quant(&args.bundle, dev, dtype, None)?;
    eprintln!("synaptix speak: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let opts = A8Options { seed: args.seed, ..pipe.default_options() };
    let t1 = std::time::Instant::now();
    let voice = match (args.reference.as_ref(), args.prompt_text.as_ref()) {
        (Some(r), Some(t)) => {
            let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(r)?;
            Some(pipe.voice_prompt(&pcm, sr, t)?)
        }
        (Some(_), None) => return Err("Audio8: для клона голоса нужен --prompt-text — точная расшифровка --reference".into()),
        _ => None,
    };
    let mut pcm = pipe.synthesize(&args.text, voice.as_ref(), &opts)?;
    if (args.speed - 1.0).abs() > 1e-3 {
        pcm = synaptix_audio::tempo::time_stretch(&pcm, pipe.sample_rate(), args.speed as f32);
    }
    let tm = pipe.last_timings();
    eprintln!(
        "synaptix speak: {} кадров ({:.1} кадр/с), EOS {}, кодек {:.2}s",
        tm.frames,
        tm.frames as f64 / tm.decode_s.max(1e-9),
        tm.finished,
        tm.codec_s
    );
    report(&pcm, pipe.sample_rate(), t1);
    Ok((pcm, pipe.sample_rate()))
}

fn speak_voxcpm(
    args: &SpeakArgs,
    dev: Device,
    dtype: DType,
    t0: std::time::Instant,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    if args.language.is_some() || args.instruct.is_some() || args.speed != 1.0 || args.target_tokens.is_some() {
        eprintln!("synaptix speak: --language/--instruct/--speed/--target-tokens — параметры OmniVoice, VoxCPM их игнорирует");
    }
    let pipe = VoxCpmPipeline::from_bundle(&args.bundle, dev, dtype)?;
    eprintln!("synaptix speak: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let defaults = GenerateOptions::default();
    let opts = GenerateOptions {
        cfg_value: args.cfg.unwrap_or(defaults.cfg_value),
        n_timesteps: args.steps.unwrap_or(defaults.n_timesteps),
        seed: args.seed,
        max_len: args.max_len,
        min_len: args.min_len.unwrap_or(defaults.min_len),
        retry_ratio_threshold: args.retry_ratio.unwrap_or(defaults.retry_ratio_threshold),
        streaming_prefix_len: args.streaming_prefix_len.unwrap_or(defaults.streaming_prefix_len),
        ..defaults
    };
    let t1 = std::time::Instant::now();
    let wav = match (args.reference.as_ref(), args.prompt_wav.as_ref(), args.prompt_text.as_ref()) {
        (Some(r), Some(pw), Some(pt)) => pipe.synthesize_combined(
            &args.text,
            pt,
            pw.to_str().ok_or("bad prompt-wav path")?,
            r.to_str().ok_or("bad reference path")?,
            &opts,
        )?,
        (Some(r), None, None) => pipe.synthesize_with_reference(&args.text, r.to_str().ok_or("bad reference path")?, &opts)?,
        (None, Some(pw), Some(pt)) => {
            pipe.synthesize_continuation(&args.text, pt, pw.to_str().ok_or("bad prompt-wav path")?, &opts)?
        }
        (None, None, None) => pipe.synthesize(&args.text, &opts)?,
        _ => return Err("prompt-wav and prompt-text must be provided together".into()),
    };
    report(&wav.pcm, wav.sample_rate as u32, t1);
    Ok((wav.pcm, wav.sample_rate as u32))
}

fn speak_omnivoice(
    args: &SpeakArgs,
    dev: Device,
    dtype: DType,
    t0: std::time::Instant,
) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    if args.prompt_wav.is_some() {
        eprintln!("synaptix speak: у OmniVoice клон голоса — --reference + --prompt-text; --prompt-wav игнорируется");
    }
    let pipe = if args.bundle.is_dir() {
        OmniVoicePipeline::from_unpacked(&args.bundle, dev, dtype)?
    } else {
        OmniVoicePipeline::from_syn(&args.bundle, dev, dtype)?
    };
    eprintln!("synaptix speak: loaded in {:.2}s", t0.elapsed().as_secs_f32());
    let defaults = OmniVoiceGenerationConfig::default();
    let gen = OmniVoiceGenerationConfig {
        num_step: args.steps.unwrap_or(defaults.num_step),
        guidance_scale: args.cfg.unwrap_or(defaults.guidance_scale),
        t_shift: args.t_shift.unwrap_or(defaults.t_shift),
        layer_penalty_factor: args.layer_penalty.unwrap_or(defaults.layer_penalty_factor),
        seed: Some(args.seed),
        denoise: !args.no_denoise,
        ..defaults
    };
    let prompt = match &args.reference {
        Some(r) => {
            let text = args
                .prompt_text
                .as_deref()
                .ok_or("клон голоса OmniVoice требует --prompt-text (текст референса)")?;
            Some(pipe.create_voice_clone_prompt(r, text)?)
        }
        None => None,
    };
    let lang = args.language.as_deref().filter(|l| !l.is_empty() && *l != "auto");
    let instruct = args.instruct.as_deref().filter(|s| !s.trim().is_empty());
    eprintln!(
        "synaptix speak: режим {} | язык {} | шагов {} | cfg {} | t_shift {} | скорость {}",
        match (&prompt, instruct) {
            (Some(_), Some(_)) => "клон + стиль",
            (Some(_), None) => "клон",
            (None, Some(_)) => "дизайн голоса",
            (None, None) => "авто",
        },
        lang.unwrap_or("авто"),
        gen.num_step,
        gen.guidance_scale,
        gen.t_shift,
        args.speed
    );
    let t1 = std::time::Instant::now();
    let pcm = pipe.synthesize(&args.text, prompt.as_ref(), lang, instruct, args.speed, args.target_tokens, &gen)?;
    let rate = pipe.output_sample_rate() as u32;
    report(&pcm, rate, t1);
    Ok((pcm, rate))
}

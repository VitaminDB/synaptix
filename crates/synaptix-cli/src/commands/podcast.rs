use std::path::PathBuf;

use synaptix_audio::io::write_wav_mono_f32;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_vibevoice::config::GenerationConfig;
use synaptix_tts_vibevoice::pipeline::{VibeVoicePipeline, VoiceSample};
use synaptix_tts_vibevoice::processor::plain_text_to_script;

use crate::commands::device;

pub struct PodcastArgs {
    pub bundle: PathBuf,
    pub script: Option<String>,
    pub script_file: Option<PathBuf>,
    pub output: PathBuf,
    pub voices: Vec<PathBuf>,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub cfg: f32,
    pub steps: usize,
    pub seed: u64,
    pub max_length_times: f32,
    pub zero_noise: bool,
}

pub fn run(args: PodcastArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.bundle.exists() {
        return Err(format!("bundle not found: {}", args.bundle.display()).into());
    }
    let raw = match (&args.script, &args.script_file) {
        (Some(s), None) => s.clone(),
        (None, Some(p)) => std::fs::read_to_string(p)
            .map_err(|e| format!("{}: {e}", p.display()))?,
        (Some(_), Some(_)) => return Err("укажите либо SCRIPT, либо --script-file".into()),
        (None, None) => return Err("нужен SCRIPT или --script-file".into()),
    };
    let script = plain_text_to_script(&raw);
    if script.is_empty() {
        return Err("пустой сценарий".into());
    }

    let dev = device::resolve(&args.device);
    let dtype = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        Some(other) => return Err(format!("unknown compute-dtype {other}").into()),
        None => match dev {
            Device::Cpu => DType::F32,
            _ => DType::BF16,
        },
    };

    let mut voices: Vec<VoiceSample> = Vec::with_capacity(args.voices.len());
    for p in &args.voices {
        voices.push(VoiceSample::from_path(p)?);
    }

    let t0 = std::time::Instant::now();
    eprintln!(
        "synaptix podcast: {} (compute={dtype:?}, {dev:?}, голосов={})",
        args.bundle.display(),
        voices.len()
    );
    let pipe = VibeVoicePipeline::from_syn(&args.bundle, dev, dtype)?;
    eprintln!("synaptix podcast: loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let cfg = GenerationConfig {
        cfg_scale: args.cfg,
        ddpm_inference_steps: args.steps,
        max_length_times: args.max_length_times,
        seed: args.seed,
        zero_noise: args.zero_noise,
        ..GenerationConfig::default()
    };

    let t1 = std::time::Instant::now();
    let mut last_pct = 0usize;
    let mut on_step = |step: usize, total: usize| {
        let pct = step * 100 / total.max(1);
        if pct >= last_pct + 5 {
            last_pct = pct;
            eprint!("\r  генерация: {pct}% ({step}/{total} шагов)");
            use std::io::Write;
            let _ = std::io::stderr().flush();
        }
    };
    let out = pipe.synthesize_with(&script, &voices, &cfg, None, Some(&mut on_step))?;
    eprintln!();

    let infer = t1.elapsed().as_secs_f32();
    let rate = pipe.sample_rate();
    let dur = out.audio.len() as f32 / rate as f32;
    eprintln!(
        "synaptix podcast: {:.2}s аудио за {:.2}s (RTF={:.3}), токенов={}{}",
        dur,
        infer,
        infer / dur.max(1e-6),
        out.tokens.len(),
        if out.reached_max { ", достигнут лимит длины" } else { "" }
    );
    if out.audio.is_empty() {
        return Err("модель не сгенерировала аудио".into());
    }

    write_wav_mono_f32(&args.output, &out.audio, rate)?;
    eprintln!("synaptix podcast: wrote {}", args.output.display());
    Ok(())
}

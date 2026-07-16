use std::path::PathBuf;

use synaptix_audio::io::write_wav_mono_f32;
use synaptix_core::dtype::DType;
use synaptix_core::device::Device;
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
    pub cfg: f32,
    pub steps: usize,
    pub seed: u64,
    pub max_len: usize,
}

pub fn run(args: SpeakArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.bundle.exists() {
        return Err(format!("bundle not found: {}", args.bundle.display()).into());
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

    let t0 = std::time::Instant::now();
    eprintln!("synaptix speak: {} (compute={dtype:?}, {dev:?})", args.bundle.display());
    let pipe = VoxCpmPipeline::from_bundle(&args.bundle, dev, dtype)?;
    eprintln!("synaptix speak: loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let opts = GenerateOptions {
        cfg_value: args.cfg,
        n_timesteps: args.steps,
        seed: args.seed,
        max_len: args.max_len,
        ..GenerateOptions::default()
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
        (Some(r), None, None) => {
            pipe.synthesize_with_reference(&args.text, r.to_str().ok_or("bad reference path")?, &opts)?
        }
        (None, Some(pw), Some(pt)) => pipe.synthesize_continuation(
            &args.text,
            pt,
            pw.to_str().ok_or("bad prompt-wav path")?,
            &opts,
        )?,
        (None, None, None) => pipe.synthesize(&args.text, &opts)?,
        _ => return Err("prompt-wav and prompt-text must be provided together".into()),
    };
    let infer = t1.elapsed().as_secs_f32();
    let dur = wav.pcm.len() as f32 / wav.sample_rate as f32;
    eprintln!(
        "synaptix speak: {:.2}s audio in {:.2}s (RTF={:.3})",
        dur,
        infer,
        infer / dur.max(1e-6)
    );

    write_wav_mono_f32(&args.output, &wav.pcm, wav.sample_rate as u32)?;
    eprintln!("synaptix speak: wrote {}", args.output.display());
    Ok(())
}

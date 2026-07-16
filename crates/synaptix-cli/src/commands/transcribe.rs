//! `synaptix transcribe <model.syn> <audio.wav>` — ASR через нативный Whisper.

use std::path::PathBuf;

use synaptix_asr_whisper::{Task, WhisperPipeline};
use synaptix_core::dtype::DType;

use crate::commands::device;

pub struct TranscribeArgs {
    pub model: PathBuf,
    pub audio: PathBuf,
    pub language: Option<String>,
    pub task: String,
    pub device: String,
    pub compute_dtype: Option<String>,
    pub timestamps: bool,
}

pub fn run(args: TranscribeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("model not found: {}", args.model.display()).into());
    }
    if !args.audio.exists() {
        return Err(format!("audio not found: {}", args.audio.display()).into());
    }

    let dev = device::resolve(&args.device);
    let dtype = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") | None => DType::F32,
        Some(other) => return Err(format!("unknown compute-dtype {other}").into()),
    };
    let task = match args.task.as_str() {
        "translate" => Task::Translate,
        "transcribe" => Task::Transcribe,
        other => return Err(format!("unknown task {other} (transcribe|translate)").into()),
    };

    let t0 = std::time::Instant::now();
    let audio = WhisperPipeline::load_audio(&args.audio)?;
    let dur_s = audio.len() as f32 / synaptix_asr_whisper::pipeline::SR as f32;
    eprintln!(
        "synaptix transcribe: {} ({:.1}s audio, compute={dtype:?}, {dev:?})",
        args.model.display(),
        dur_s
    );

    let pipe = WhisperPipeline::from_syn(&args.model, dev, dtype)?;
    eprintln!("synaptix transcribe: model loaded in {:.2}s", t0.elapsed().as_secs_f32());

    let t1 = std::time::Instant::now();
    if args.timestamps {
        let segs = pipe.transcribe_timestamped(&audio, args.language.as_deref(), task)?;
        let infer = t1.elapsed().as_secs_f32();
        eprintln!("synaptix transcribe: done in {:.2}s (RTF={:.3})", infer, infer / dur_s.max(1e-6));
        for s in segs {
            println!("[{:>7.2} -> {:>7.2}] {}", s.start, s.end, s.text);
        }
    } else {
        let text = pipe.transcribe(&audio, args.language.as_deref(), task)?;
        let infer = t1.elapsed().as_secs_f32();
        eprintln!("synaptix transcribe: done in {:.2}s (RTF={:.3})", infer, infer / dur_s.max(1e-6));
        println!("{text}");
    }
    Ok(())
}

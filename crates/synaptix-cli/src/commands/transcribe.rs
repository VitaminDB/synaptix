use std::path::{Path, PathBuf};

use synaptix_asr_whisper::{Task, WhisperPipeline};
use synaptix_core::device::Device;
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
    pub engine: String,
    pub format: String,
    pub output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Engine {
    Whisper,
    GigaAm,
}

struct Segment {
    start: f32,
    end: f32,
    text: String,
}

fn detect_engine(model: &Path, requested: &str) -> Result<Engine, String> {
    match requested {
        "whisper" => return Ok(Engine::Whisper),
        "gigaam" => return Ok(Engine::GigaAm),
        "auto" => {}
        o => return Err(format!("--engine: auto | whisper | gigaam, а не `{o}`")),
    }
    let id = synaptix_bundle::Bundle::open(model)
        .map(|b| b.id().to_lowercase())
        .unwrap_or_default();
    let name = model.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    Ok(if id.contains("gigaam") || name.contains("gigaam") { Engine::GigaAm } else { Engine::Whisper })
}

pub fn load_audio_16k(path: &Path) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match crate::commands::sheet::load_audio(path, 16_000) {
        Ok(pcm) => Ok(pcm),
        Err(_) => Ok(WhisperPipeline::load_audio(path)?),
    }
}

fn clock(t: f32, sep: char) -> String {
    let ms = (t.max(0.0) * 1000.0).round() as u64;
    format!("{:02}:{:02}:{:02}{sep}{:03}", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, ms % 1000)
}

fn render(format: &str, text: &str, segments: &[Segment], language: Option<&str>) -> Result<String, String> {
    Ok(match format {
        "text" => {
            if segments.is_empty() {
                format!("{text}\n")
            } else {
                segments.iter().map(|s| format!("[{:>7.2} -> {:>7.2}] {}\n", s.start, s.end, s.text)).collect()
            }
        }
        "srt" => segments
            .iter()
            .enumerate()
            .map(|(i, s)| format!("{}\n{} --> {}\n{}\n\n", i + 1, clock(s.start, ','), clock(s.end, ','), s.text.trim()))
            .collect(),
        "vtt" => {
            let body: String = segments
                .iter()
                .map(|s| format!("{} --> {}\n{}\n\n", clock(s.start, '.'), clock(s.end, '.'), s.text.trim()))
                .collect();
            format!("WEBVTT\n\n{body}")
        }
        "json" => {
            let segs: Vec<serde_json::Value> = segments
                .iter()
                .map(|s| serde_json::json!({"start": s.start, "end": s.end, "text": s.text.trim()}))
                .collect();
            let mut v = serde_json::json!({"text": text, "segments": segs});
            if let Some(l) = language {
                v["language"] = serde_json::Value::String(l.to_string());
            }
            serde_json::to_string_pretty(&v).map_err(|e| e.to_string())? + "\n"
        }
        o => return Err(format!("--format: text | srt | vtt | json, а не `{o}`")),
    })
}

pub fn run(args: TranscribeArgs) -> Result<(), Box<dyn std::error::Error>> {
    if !args.model.exists() {
        return Err(format!("model not found: {}", args.model.display()).into());
    }
    if !args.audio.exists() {
        return Err(format!("audio not found: {}", args.audio.display()).into());
    }
    let engine = detect_engine(&args.model, &args.engine)?;
    let dev = device::resolve(&args.device);
    let dtype = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        None if dev == Device::Cpu => DType::F32,
        None => DType::F16,
        Some(other) => return Err(format!("unknown compute-dtype {other}").into()),
    };
    let task = match args.task.as_str() {
        "translate" => Task::Translate,
        "transcribe" => Task::Transcribe,
        other => return Err(format!("unknown task {other} (transcribe|translate)").into()),
    };
    let need_segments = args.timestamps || matches!(args.format.as_str(), "srt" | "vtt");
    let t0 = std::time::Instant::now();
    let audio = load_audio_16k(&args.audio)?;
    let dur_s = audio.len() as f32 / 16_000.0;
    eprintln!(
        "synaptix transcribe: {} [{engine:?}] ({:.1}s audio, compute={dtype:?}, {dev:?})",
        args.model.display(),
        dur_s
    );
    let (text, segments, language) = match engine {
        Engine::GigaAm => {
            if args.language.as_deref().is_some_and(|l| l != "ru") || matches!(task, Task::Translate) {
                return Err("GigaAM — только русская транскрибация (--language ru, --task transcribe)".into());
            }
            if need_segments {
                return Err("у GigaAM нет временных меток: --format text | json без --timestamps".into());
            }
            let pipe = if args.model.is_dir() {
                synaptix_asr_gigaam::GigaAm::from_unpacked(&args.model, &dev, dtype)?
            } else {
                synaptix_asr_gigaam::GigaAm::from_syn(&args.model, &dev, dtype)?
            };
            eprintln!("synaptix transcribe: model loaded in {:.2}s", t0.elapsed().as_secs_f32());
            let t1 = std::time::Instant::now();
            let text = pipe.transcribe_pcm(&audio, 16_000)?;
            let infer = t1.elapsed().as_secs_f32();
            eprintln!("synaptix transcribe: done in {:.2}s (RTF={:.3})", infer, infer / dur_s.max(1e-6));
            (text, Vec::new(), Some("ru".to_string()))
        }
        Engine::Whisper => {
            let pipe = WhisperPipeline::from_syn(&args.model, dev, dtype)?;
            eprintln!("synaptix transcribe: model loaded in {:.2}s", t0.elapsed().as_secs_f32());
            let language = match &args.language {
                Some(l) => l.clone(),
                None => {
                    let l = pipe.detect_language_code(&audio)?;
                    eprintln!("synaptix transcribe: язык — {l}");
                    l
                }
            };
            let t1 = std::time::Instant::now();
            let (text, segments) = if need_segments {
                let segs = pipe.transcribe_timestamped(&audio, Some(&language), task)?;
                let text = segs.iter().map(|s| s.text.trim()).collect::<Vec<_>>().join(" ");
                let segs = segs.into_iter().map(|s| Segment { start: s.start, end: s.end, text: s.text }).collect();
                (text, segs)
            } else {
                (pipe.transcribe(&audio, Some(&language), task)?, Vec::new())
            };
            let infer = t1.elapsed().as_secs_f32();
            eprintln!("synaptix transcribe: done in {:.2}s (RTF={:.3})", infer, infer / dur_s.max(1e-6));
            (text, segments, Some(language))
        }
    };
    let out = render(&args.format, &text, &segments, language.as_deref())?;
    match &args.output {
        Some(p) => {
            std::fs::write(p, out)?;
            eprintln!("synaptix transcribe: wrote {}", p.display());
        }
        None => print!("{out}"),
    }
    Ok(())
}

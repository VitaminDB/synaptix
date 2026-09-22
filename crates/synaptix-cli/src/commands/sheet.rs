//! `synaptix sheet` — запись → партитура ABC (SheetSage2), и `synaptix
//! sheet-pack` — упаковка релиза SheetSage2 компонентом в бандл YuE2.

use std::io::Write;
use std::path::{Path, PathBuf};

use synaptix_core::dtype::DType;
use synaptix_music_sheetsage2::pipeline::{prepare_audio, Progress};
use synaptix_music_sheetsage2::{SheetSage2, TranscribeOptions, Transcription, VoiceSelect};
use synaptix_music_yue2::pipeline::{find_bundle, MODEL_NAMES};

use crate::commands::device;

pub struct SheetArgs {
    pub audio: PathBuf,
    pub output: PathBuf,
    pub models: PathBuf,
    pub model: Option<PathBuf>,
    /// С аккордами (по умолчанию — только мелодия, для каверов).
    pub full: bool,
    /// `both` | `vocal` | `ins`.
    pub voices: String,
    pub max_seconds: Option<f64>,
    pub device: String,
    pub compute_dtype: Option<String>,
    /// Сохранить токены окон (для сверки с релизом).
    pub tokens_json: Option<PathBuf>,
}

pub struct SheetPackArgs {
    pub sheetsage: PathBuf,
    pub mert: Option<PathBuf>,
    pub into: PathBuf,
}

pub fn parse_voices(s: &str) -> Result<VoiceSelect, String> {
    match s {
        "both" => Ok(VoiceSelect::Both),
        "vocal" => Ok(VoiceSelect::Vocal),
        "ins" | "instrumental" => Ok(VoiceSelect::Instrumental),
        other => Err(format!("--voices ожидает both | vocal | ins, а не `{other}`")),
    }
}

/// Звук → моно 24 кГц. Как релиз: через ffmpeg (`-ac 1 -ar 24000`), а без
/// него — WAV и свой ресэмплинг.
pub fn load_audio(path: &Path, rate: u32) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let out = std::process::Command::new("ffmpeg")
        .args(["-v", "error", "-nostdin", "-i"])
        .arg(path)
        .args(["-vn", "-ac", "1", "-ar", &rate.to_string(), "-f", "f32le", "pipe:1"])
        .output();
    if let Ok(out) = out {
        if out.status.success() {
            return Ok(out.stdout.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect());
        }
        return Err(format!("ffmpeg: {}", String::from_utf8_lossy(&out.stderr)).into());
    }
    let (interleaved, sr) = synaptix_audio::io::read_wav_stereo_f32(path)?;
    let left: Vec<f32> = interleaved.iter().step_by(2).copied().collect();
    let right: Vec<f32> = interleaved.iter().skip(1).step_by(2).copied().collect();
    Ok(prepare_audio(&[left, right], sr, rate)?)
}

pub fn resolve_model(models: &Path, model: Option<PathBuf>) -> Result<PathBuf, String> {
    match model {
        Some(p) => Ok(p),
        None => find_bundle(models, MODEL_NAMES)
            .ok_or_else(|| format!("в {} нет ни одного из {MODEL_NAMES:?} — укажите --model", models.display())),
    }
}

/// Транскрибировать файл с выводом прогресса в stderr.
pub fn transcribe_file(
    model: &Path,
    audio: &Path,
    device: &str,
    compute: DType,
    options: &TranscribeOptions,
) -> Result<Transcription, Box<dyn std::error::Error>> {
    let dev = device::resolve(device);
    let sheet = SheetSage2::open(model, dev, compute)?;
    eprintln!(
        "[sheetsage2] {} | {:?} {:?} | загрузка {:.1} с",
        model.display(),
        dev,
        compute,
        sheet.load_seconds
    );
    let samples = load_audio(audio, sheet.sample_rate())?;
    eprintln!(
        "[sheetsage2] {} — {:.1} с звука",
        audio.display(),
        samples.len() as f64 / sheet.sample_rate() as f64
    );
    let mut progress = |p: Progress| {
        match p {
            Progress::Encoding { window, windows } => eprint!("\r[sheetsage2] окно {window}/{windows}: энкодер     "),
            Progress::Decoding { window, windows, tokens } => {
                eprint!("\r[sheetsage2] окно {window}/{windows}: {tokens} токенов     ")
            }
            Progress::Notation => eprint!("\r[sheetsage2] нотация                    "),
        }
        let _ = std::io::stderr().flush();
    };
    let result = sheet.transcribe(&samples, options, &mut progress, &|| false)?;
    eprintln!();
    for (i, w) in result.windows.iter().enumerate() {
        eprintln!(
            "[sheetsage2] окно {}: {:.0}–{:.0} с, префикс {} ток., {} ток., событий {} (принято {}), энкодер {:.2} с, декодер {:.2} с ({:.0} ток/с)",
            i + 1,
            w.window.start,
            w.window.end,
            w.prefix_tokens,
            w.tokens.len(),
            w.events,
            w.accepted_events,
            w.encode_seconds,
            w.decode_seconds,
            (w.tokens.len() - w.prefix_tokens) as f64 / w.decode_seconds.max(1e-9)
        );
    }
    for warning in &result.warnings {
        eprintln!("[sheetsage2] предупреждение: {warning}");
    }
    Ok(result)
}

pub fn run(args: SheetArgs) -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let model = resolve_model(&args.models, args.model)?;
    let compute = match args.compute_dtype.as_deref() {
        Some("f32") | Some("fp32") => DType::F32,
        Some("bf16") | None => DType::BF16,
        Some(other) => return Err(format!("--compute-dtype: bf16 | f32, а не `{other}`").into()),
    };
    let options = TranscribeOptions {
        melody_only: !args.full,
        voices: parse_voices(&args.voices)?,
        max_seconds: args.max_seconds,
        ..Default::default()
    };
    let result = transcribe_file(&model, &args.audio, &args.device, compute, &options)?;
    if let Some(path) = &args.tokens_json {
        let windows: Vec<serde_json::Value> = result
            .windows
            .iter()
            .map(|w| {
                serde_json::json!({
                    "start": w.window.start, "end": w.window.end,
                    "prefix_tokens": w.prefix_tokens, "tokens": w.tokens,
                })
            })
            .collect();
        std::fs::write(path, serde_json::to_vec(&windows)?)?;
    }
    let e = &result.export;
    eprintln!(
        "[sheetsage2] нот {} (вокал {}, инструмент {}), тактов {}, всего {:.1} с",
        e.melody_notes, e.vocal_notes, e.instrumental_notes, e.measures, result.seconds
    );
    match (&result.abc, &result.abc_error) {
        (Some(abc), _) => {
            std::fs::write(&args.output, abc)?;
            eprintln!("[sheetsage2] партитура → {}", args.output.display());
            Ok(())
        }
        (None, Some(err)) => Err(format!("партитура не собрана: {err}").into()),
        (None, None) => Err("партитура не собрана".into()),
    }
}

pub fn run_pack(args: SheetPackArgs) -> Result<(), Box<dyn std::error::Error>> {
    let t0 = std::time::Instant::now();
    let mut progress = |msg: &str| eprintln!("[sheet-pack] {msg}");
    let stats = synaptix_music_sheetsage2::pack::pack_into_bundle(
        &args.into,
        &args.sheetsage,
        args.mert.as_deref(),
        &mut progress,
    )?;
    eprintln!(
        "[sheet-pack] {} ← компонент sheetsage2: {} тензоров ({} в BF16), влито LoRA {}, {:.2} ГБ, {:.1} с",
        args.into.display(),
        stats.tensors,
        stats.bf16_tensors,
        stats.merged,
        stats.payload_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

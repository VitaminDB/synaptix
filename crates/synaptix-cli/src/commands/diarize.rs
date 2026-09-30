use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_diarization_sortformer::{DiarizationResult, SortformerPipeline};

use crate::commands::device;

pub struct DiarizeArgs {
    pub model: PathBuf,
    pub audio: PathBuf,
    pub output: Option<PathBuf>,
    pub format: String,
    pub threshold: Option<f32>,
    pub min_segment: Option<f32>,
    pub merge_gap: Option<f32>,
    pub smoothing_frames: Option<usize>,
    pub device: String,
    pub compute_dtype: Option<String>,
}

pub fn run(args: DiarizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dev = device::resolve(&args.device);
    let dtype = match args.compute_dtype.as_deref() {
        Some("f16") => DType::F16,
        Some("bf16") => DType::BF16,
        Some("f32") => DType::F32,
        None if dev == Device::Cpu => DType::F32,
        None => DType::F16,
        Some(o) => return Err(format!("unknown compute-dtype {o}").into()),
    };
    let t0 = std::time::Instant::now();
    let pcm = crate::commands::transcribe::load_audio_16k(&args.audio)?;
    let pipe = SortformerPipeline::from_syn(&args.model, dev, dtype)?;
    let mut params = pipe.default_params();
    if let Some(v) = args.threshold {
        params.threshold = v;
    }
    if let Some(v) = args.min_segment {
        params.min_segment_s = v;
    }
    if let Some(v) = args.merge_gap {
        params.merge_gap_s = v;
    }
    if let Some(v) = args.smoothing_frames {
        params.smoothing_frames = v.max(1) | 1;
    }
    let segments = pipe.diarize_with(&pcm, 16_000, &params)?;
    let duration_s = pcm.len() as f32 / 16_000.0;
    let mut speakers: Vec<u8> = segments.iter().map(|s| s.speaker).collect();
    speakers.sort_unstable();
    speakers.dedup();
    let result = DiarizationResult {
        version: synaptix_diarization_sortformer::postprocess::RESULT_VERSION,
        sample_rate: 16_000,
        duration_s,
        num_speakers: speakers.len(),
        segments,
    };
    eprintln!(
        "synaptix diarize: {:.1}s audio, {} спикеров, {} сегментов за {:.2}s",
        duration_s,
        result.num_speakers,
        result.segments.len(),
        t0.elapsed().as_secs_f32()
    );
    let stem = args.audio.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "audio".into());
    let out = match args.format.as_str() {
        "pretty" => result
            .segments
            .iter()
            .map(|s| format!("[{:>8.2} → {:>8.2}] спикер {} ({:.2})\n", s.start_s, s.end_s, s.speaker + 1, s.confidence))
            .collect::<String>(),
        "json" => serde_json::to_string_pretty(&result)? + "\n",
        "rttm" => result
            .segments
            .iter()
            .map(|s| format!("SPEAKER {stem} 1 {:.3} {:.3} <NA> <NA> speaker_{} <NA> <NA>\n", s.start_s, s.end_s - s.start_s, s.speaker))
            .collect(),
        o => return Err(format!("--format: pretty | json | rttm, а не `{o}`").into()),
    };
    match &args.output {
        Some(p) => {
            std::fs::write(p, out)?;
            eprintln!("synaptix diarize: wrote {}", p.display());
        }
        None => print!("{out}"),
    }
    Ok(())
}

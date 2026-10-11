use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::quant::calib;

pub struct CalibrateArgs {
    pub model: PathBuf,
    pub audio: PathBuf,
    pub output: PathBuf,
    pub device: String,
    pub limit: usize,
}

fn wavs(dir: &Path, limit: usize) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("wav")) {
                out.push(p);
            }
        }
    }
    out.sort();
    if out.len() > limit {
        let step = out.len() as f64 / limit as f64;
        out = (0..limit).map(|i| out[(i as f64 * step) as usize].clone()).collect();
    }
    out
}

fn is_omnivoice(model: &Path) -> bool {
    synaptix_bundle::Bundle::open(model)
        .map(|b| b.tensors_slice_named("lm").is_ok() && b.tensors_slice_named("codec").is_ok())
        .unwrap_or(false)
}

fn tts_lines(path: &Path, limit: usize) -> Result<Vec<(Option<String>, String)>, Box<dyn std::error::Error>> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .take(limit)
        .map(|l| match l.split_once('|') {
            Some((lang, t)) => (Some(lang.trim().to_string()), t.trim().to_string()),
            None => (None, l.to_string()),
        })
        .collect())
}

fn calibrate_tts(args: &CalibrateArgs, device: Device) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_tts_omnivoice::{OmniVoiceGenerationConfig, OmniVoicePipeline};
    let lines = tts_lines(&args.audio, args.limit)?;
    if lines.is_empty() {
        return Err(format!("{}: нет фраз (строки «язык|текст»)", args.audio.display()).into());
    }
    let dtype = if matches!(device, Device::Npu(_)) { DType::F16 } else { DType::F32 };
    let tts = OmniVoicePipeline::from_syn(&args.model, device, dtype)?;
    let gen = OmniVoiceGenerationConfig::default();
    println!("Calibrate {} (TTS) на {} фразах ({:?})", args.model.display(), lines.len(), device);
    calib::start();
    let t0 = std::time::Instant::now();
    for (i, (lang, text)) in lines.iter().enumerate() {
        tts.synthesize(text, None, lang.as_deref(), None, 1.0, None, &gen)?;
        eprint!("\r  [{}/{}]", i + 1, lines.len());
    }
    eprintln!();
    let stats = calib::finish();
    calib::save(&args.output, &stats)?;
    println!("Done: {} ({} слоёв, {:.1} с)", args.output.display(), stats.len(), t0.elapsed().as_secs_f32());
    Ok(())
}

pub fn run(args: CalibrateArgs) -> Result<(), Box<dyn std::error::Error>> {
    synaptix::init()?;
    let device = super::device::resolve(&args.device);
    if is_omnivoice(&args.model) {
        return calibrate_tts(&args, device);
    }
    let files = if args.audio.is_dir() { wavs(&args.audio, args.limit) } else { vec![args.audio.clone()] };
    if files.is_empty() {
        return Err(format!("{}: нет WAV-файлов", args.audio.display()).into());
    }
    let dtype = if matches!(device, Device::Npu(_)) { DType::F16 } else { DType::F32 };
    let model = synaptix_asr_wav2vec2::Wav2Vec2Ctc::load(&args.model, &device, dtype)?;
    println!("Calibrate {} на {} записях ({:?})", args.model.display(), files.len(), device);
    calib::start();
    let t0 = std::time::Instant::now();
    for (i, f) in files.iter().enumerate() {
        let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(f)?;
        model.log_probs(&pcm, sr)?;
        eprint!("\r  [{}/{}] {}", i + 1, files.len(), f.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
    }
    eprintln!();
    let stats = calib::finish();
    calib::save(&args.output, &stats)?;
    println!("Done: {} ({} слоёв, {:.1} с)", args.output.display(), stats.len(), t0.elapsed().as_secs_f32());
    Ok(())
}

fn warmup_tts(model: &Path, device: Device) -> Result<(), Box<dyn std::error::Error>> {
    use synaptix_tts_omnivoice::OmniVoicePipeline;
    let dtype = if matches!(device, Device::Npu(_)) { DType::F16 } else { DType::F32 };
    let t0 = std::time::Instant::now();
    let tts = OmniVoicePipeline::from_syn(model, device, dtype)?;
    println!("Warmup {} (TTS, {:?}), загрузка {:.1} с", model.display(), device, t0.elapsed().as_secs_f32());
    let n_cb = tts.config().num_audio_codebook;
    for s in synaptix_tts_omnivoice::backbone::NPU_BUCKETS.iter().map(|&b| b - 1) {
        let ids = vec![0i64; n_cb * s];
        let mask = vec![1u8; s];
        let t = std::time::Instant::now();
        let sb = tts.backbone().bucket_for(s).unwrap_or(s);
        tts.backbone().forward_padded(&[(&ids, &mask, s)], sb)?.to_dtype(DType::F32)?.to_device(Device::Cpu)?;
        println!("  {s} позиций: {:.1} с", t.elapsed().as_secs_f32());
    }
    Ok(())
}

pub fn warmup(model: &Path, device: &str) -> Result<(), Box<dyn std::error::Error>> {
    synaptix::init()?;
    let device = super::device::resolve(device);
    if is_omnivoice(model) {
        return warmup_tts(model, device);
    }
    let dtype = if device == Device::Cpu { DType::F32 } else { DType::F16 };
    let t0 = std::time::Instant::now();
    let m = synaptix_asr_wav2vec2::Wav2Vec2Ctc::load(model, &device, dtype)?;
    println!("Warmup {} ({:?}), загрузка {:.1} с", model.display(), device, t0.elapsed().as_secs_f32());
    let buckets: Vec<usize> = if m.length_buckets_active() { synaptix_asr_wav2vec2::Wav2Vec2Ctc::length_buckets().to_vec() } else { vec![32000] };
    let mut x = 12345u32;
    for n in buckets {
        let pcm: Vec<f32> = (0..n.saturating_sub(1))
            .map(|_| {
                x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((x >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.02
            })
            .collect();
        let t = std::time::Instant::now();
        m.log_probs(&pcm, 16000)?;
        m.log_probs_hidden(&pcm, 16000)?;
        println!("  {:.0} с записи: {:.1} с", n as f32 / 16000.0, t.elapsed().as_secs_f32());
    }
    Ok(())
}

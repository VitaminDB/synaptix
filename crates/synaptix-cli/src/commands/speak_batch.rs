//! Пакетный синтез Silero: jsonl → WAV + meta.jsonl (символы и их длительности — разметка без выравнивателя).
//!
//! Вход — строки `{"id","text","voice","english","rate","pitch","accents"}`
//! (`accents: false` — без автоматических ударений и омографов: только «+» из текста). Параллелизм — по фразам: воркеры
//! (потоки) с общей моделью; сеть маленькая (~23M), батч 1 упирается в запуски ядер, поэтому
//! пропускная способность набирается числом одновременно синтезируемых фраз.

use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use serde::Deserialize;
use synaptix_tts_silero::{Options, SileroPipeline};

use crate::commands::device;

pub struct BatchArgs {
    pub bundle: PathBuf,
    pub input: Option<PathBuf>,
    pub out_dir: Option<PathBuf>,
    pub device: String,
    pub workers: usize,
    pub sr: u32,
    pub list_voices: bool,
    pub skip_existing: bool,
}

#[derive(Deserialize)]
struct Item {
    id: String,
    text: String,
    voice: Option<String>,
    english: Option<bool>,
    rate: Option<f32>,
    pitch: Option<f32>,
    accents: Option<bool>,
}

fn write_wav16(path: &Path, pcm: &[f32], sr: u32) -> std::io::Result<()> {
    let n = (pcm.len() * 2) as u32;
    let mut w = Vec::with_capacity(44 + pcm.len() * 2);
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + n).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    for v in [16u32] {
        w.extend_from_slice(&v.to_le_bytes());
    }
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&sr.to_le_bytes());
    w.extend_from_slice(&(sr * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&n.to_le_bytes());
    for v in pcm {
        w.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0).round() as i16).to_le_bytes());
    }
    std::fs::write(path, w)
}

pub fn run(a: BatchArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dev = device::resolve(&a.device);
    let t0 = std::time::Instant::now();
    let pipe = Arc::new(SileroPipeline::from_syn(&a.bundle, dev)?);
    if a.list_voices {
        for v in pipe.speakers() {
            println!("{v}");
        }
        return Ok(());
    }
    let input = a.input.ok_or("нужен --input in.jsonl")?;
    let out = a.out_dir.ok_or("нужен --out-dir")?;
    std::fs::create_dir_all(&out)?;
    let items: Vec<Item> = std::io::BufReader::new(std::fs::File::open(&input)?)
        .lines()
        .map_while(|l| l.ok())
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<Item>(&l).map_err(|e| format!("{}: {e}: {l}", input.display())))
        .collect::<Result<_, _>>()?;
    let speakers = pipe.speakers();
    let default_voice = speakers.iter().find(|s| *s == "xenia").cloned().unwrap_or_else(|| speakers[0].clone());
    eprintln!("speak-batch: {} фраз, модель {:.2} с, устройство {dev:?}, воркеров {}", items.len(), t0.elapsed().as_secs_f32(), a.workers);
    // Silero: 48 кГц (родная), 24/8 — PQMF; прочие (16 кГц) — ресемпл из 48
    let native = if [48_000, 24_000, 8_000].contains(&a.sr) { a.sr } else { 48_000 };
    let meta = Arc::new(Mutex::new(std::fs::OpenOptions::new().create(true).append(true).open(out.join("meta.jsonl"))?));
    let next = Arc::new(AtomicUsize::new(0));
    let done_s = Arc::new(Mutex::new(0f64));
    let errors = Arc::new(AtomicUsize::new(0));
    let items = Arc::new(items);
    let t1 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for _ in 0..a.workers.max(1) {
            let (pipe, items, next, meta, done_s, errors, out) = (pipe.clone(), items.clone(), next.clone(), meta.clone(), done_s.clone(), errors.clone(), out.clone());
            let default_voice = default_voice.clone();
            let skip = a.skip_existing;
            let sr = a.sr;
            sc.spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed);
                let Some(it) = items.get(i) else { break };
                let wav = out.join(format!("{}.wav", it.id));
                if skip && wav.exists() {
                    continue;
                }
                let o = Options {
                    speaker: it.voice.clone().unwrap_or_else(|| default_voice.clone()),
                    sample_rate: native,
                    rate: it.rate.unwrap_or(1.0),
                    pitch: it.pitch.unwrap_or(1.0),
                    english: it.english,
                    accents: it.accents.unwrap_or(true),
                    ..Options::default()
                };
                let r = pipe.infer_marked(&it.text, &o).map_err(|e| e.to_string()).and_then(|(syms, net)| {
                    let pcm = if sr == native { net.audio } else { synaptix_audio::resample::resample(&net.audio, native, sr).map_err(|e| e.to_string())? };
                    write_wav16(&wav, &pcm, sr).map_err(|e| e.to_string())?;
                    let secs = pcm.len() as f64 / sr as f64;
                    let durs: Vec<f64> = net.dur.iter().map(|&d| (d as f64 * SileroPipeline::FRAME_S as f64 * 1e4).round() / 1e4).collect();
                    let line = serde_json::json!({"id": it.id, "voice": o.speaker, "text": it.text, "symbols": syms, "durations": durs, "seconds": secs, "sample_rate": sr});
                    if let Ok(mut m) = meta.lock() {
                        let _ = writeln!(m, "{line}");
                    }
                    Ok(secs)
                });
                match r {
                    Ok(s) => *done_s.lock().unwrap() += s,
                    Err(e) => {
                        errors.fetch_add(1, Ordering::Relaxed);
                        eprintln!("  {}: ошибка: {e}", it.id);
                    }
                }
                if i % 1000 == 999 {
                    let s = *done_s.lock().unwrap();
                    let el = t1.elapsed().as_secs_f64();
                    eprintln!("  {}/{}: {:.2} ч звука за {:.1} мин ({:.2} ч/мин)", i + 1, items.len(), s / 3600.0, el / 60.0, s / 3600.0 / (el / 60.0));
                }
            });
        }
    });
    let s = *done_s.lock().unwrap();
    let el = t1.elapsed().as_secs_f64();
    eprintln!(
        "speak-batch: {:.2} ч звука за {:.1} с — {:.2} ч звука в минуту (RTF {:.5}), ошибок {}",
        s / 3600.0,
        el,
        s / 3600.0 / (el / 60.0),
        el / s.max(1e-9),
        errors.load(Ordering::Relaxed)
    );
    Ok(())
}

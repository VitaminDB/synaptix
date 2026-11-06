//! Сверка с эталоном `reference/f5_ref.py` и замер скорости.
//!
//! f5_audit <bundle.syn> --ref-wav ref.wav --ref-text "..." --text "..." [--refdir DIR]
//!     [--device cuda|cpu] [--dtype f32|f16|bf16] [--nfe 32] [--seed 0] [--speed 1] [--out out.wav]
//!     [--prepared] (образец без обрезки тишины — как эталон)

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_f5::{F5Pipeline, Options, VoicePrompt};

fn npy(path: &Path) -> (Vec<usize>, String, Vec<u8>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = String::from_utf8_lossy(&b[off..off + hl]).to_string();
    let descr = h.split("'descr':").nth(1).unwrap().split('\'').nth(1).unwrap().to_string();
    let s = h.split("'shape':").nth(1).unwrap();
    let s = &s[s.find('(').unwrap() + 1..s.find(')').unwrap()];
    (s.split(',').filter_map(|x| x.trim().parse().ok()).collect(), descr, b[off + hl..].to_vec())
}

fn f32s(p: &Path) -> Vec<f32> {
    let (_, d, b) = npy(p);
    assert_eq!(d, "<f4");
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (mut d2, mut b2) = (0f64, 0f64);
    for i in 0..n {
        d2 += ((a[i] - b[i]) as f64).powi(2);
        b2 += (b[i] as f64).powi(2);
    }
    (d2 / b2.max(1e-30)).sqrt() as f32
}

fn write_wav(path: &str, pcm: &[f32], sr: u32) {
    let mut w = Vec::with_capacity(44 + pcm.len() * 2);
    let n = (pcm.len() * 2) as u32;
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + n).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes());
    w.extend_from_slice(&sr.to_le_bytes());
    w.extend_from_slice(&(sr * 2).to_le_bytes());
    w.extend_from_slice(&2u16.to_le_bytes());
    w.extend_from_slice(&16u16.to_le_bytes());
    w.extend_from_slice(b"data");
    w.extend_from_slice(&n.to_le_bytes());
    for v in pcm {
        w.extend_from_slice(&((v.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, w).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let device = if get("--device").as_deref() == Some("cpu") { Device::Cpu } else { Device::Cuda(0) };
    synaptix_kernels_cpu::ensure_registered();
    if device.is_cuda() {
        synaptix_kernels_cuda::ensure_registered();
    }
    let dtype = get("--dtype").map(|s| match s.as_str() {
        "f16" => DType::F16,
        "bf16" => DType::BF16,
        _ => DType::F32,
    });
    let t = Instant::now();
    let p = F5Pipeline::from_syn(Path::new(&a[1]), device, dtype, None).expect("load");
    eprintln!("загрузка {:.2} с: {}", t.elapsed().as_secs_f64(), p.config.get("description").and_then(|v| v.as_str()).unwrap_or("?"));
    let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(get("--ref-wav").expect("--ref-wav")).expect("ref wav");
    let ref_text = get("--ref-text").expect("--ref-text");
    let voice = if a.iter().any(|x| x == "--prepared") {
        assert_eq!(sr, 24_000);
        VoicePrompt::prepared(pcm, &ref_text)
    } else {
        VoicePrompt::new(&pcm, sr, &ref_text).unwrap()
    };
    let text = get("--text").expect("--text");
    let o = Options {
        nfe: get("--nfe").map(|s| s.parse().unwrap()).unwrap_or(32),
        seed: get("--seed").map(|s| s.parse().unwrap()).unwrap_or(0),
        speed: get("--speed").map(|s| s.parse().unwrap()).unwrap_or(1.0),
        ..Options::default()
    };
    if let Some(r) = get("--refdir").map(PathBuf::from) {
        let mut rt = voice.text.clone();
        if rt.chars().last().is_some_and(|c| c.len_utf8() == 1) && !rt.ends_with(' ') {
            rt.push(' ');
        }
        let ids = p.text_ids(&rt, &text);
        let (_, _, rb) = npy(&r.join("text_ids.npy"));
        let rids: Vec<i64> = rb.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
        eprintln!("ids: {} (эталон {}), {}", ids.len(), rids.len(), if ids == rids { "совпали" } else { "РАЗНЫЕ" });
        if ids != rids {
            let d: Vec<(usize, i64, i64)> = ids.iter().zip(&rids).enumerate().filter(|(_, (x, y))| x != y).map(|(i, (x, y))| (i, *x, *y)).take(10).collect();
            eprintln!("  расхождения (позиция, наш, эталон): {d:?}");
        }
        let rm = f32s(&r.join("ref_mel.npy"));
        let om: Vec<f32> = F5Pipeline::ref_mel(&voice).into_iter().flatten().collect();
        eprintln!("мел образца: {} vs {} значений, отн. ошибка {:.2e}", om.len(), rm.len(), rel(&om, &rm));
        let dur = std::fs::read_to_string(r.join("duration.txt")).unwrap();
        let t = Instant::now();
        let (mel, n) = p.sample(&voice, &text, &o).unwrap();
        eprintln!("DiT: {:.2} с; длительность {n} кадров (эталон «{}»)", t.elapsed().as_secs_f64(), dur.trim());
        let mf: Vec<f32> = mel.iter().flatten().copied().collect();
        let rmel = f32s(&r.join("mel.npy"));
        eprintln!("мел: {} vs {} значений, отн. ошибка {:.2e}", mf.len(), rmel.len(), rel(&mf, &rmel));
        // вокодер на мел эталона — отдельно
        let rmel_rows: Vec<Vec<f32>> = rmel.chunks(100).map(|c| c.to_vec()).collect();
        let w = p.vocode(&rmel_rows).unwrap();
        let (rw, _) = synaptix_audio::io::read_wav_mono_f32(r.join("audio.wav")).unwrap();
        let scale = if voice.rms < 0.1 { voice.rms / 0.1 } else { 1.0 };
        let w: Vec<f32> = w.iter().map(|v| v * scale).collect();
        eprintln!("вокодер (мел эталона): {} vs {} семплов, отн. ошибка {:.2e}", w.len(), rw.len(), rel(&w, &rw));
    }
    let t = Instant::now();
    let out = p.synthesize(&voice, &text, &o).unwrap();
    let el = t.elapsed().as_secs_f64();
    let tm = p.last_timings();
    eprintln!("синтез: {:.2} с звука за {el:.2} с (RTF {:.3}; DiT {:.2} с, вокодер {:.2} с, кусков {})", tm.audio_s, el / tm.audio_s.max(1e-9), tm.dit_s, tm.vocoder_s, tm.chunks);
    if let Some(o) = get("--out") {
        write_wav(&o, &out, 24_000);
        eprintln!("→ {o}");
    }
}

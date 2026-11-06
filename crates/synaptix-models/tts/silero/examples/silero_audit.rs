//! Сверка с эталоном `reference/silero_ref.py` и замер скорости.
//!
//! silero_audit <bundle.syn> --speaker xenia --text "..." [--refdir DIR] [--sr 48000]
//!     [--device cpu|cuda] [--out out.wav] [--no-accent]
//! silero_audit <bundle.syn> --lines texts.txt --golden accented_lines.txt   (только ударения)

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_tts_silero::{Options, SileroPipeline};

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

fn i64s(p: &Path) -> Vec<i64> {
    let (_, d, b) = npy(p);
    assert_eq!(d, "<i8");
    b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect()
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
    for (a, b) in [(b"RIFF", 36 + n)] {
        w.extend_from_slice(a);
        w.extend_from_slice(&b.to_le_bytes());
    }
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
    let device = if get("--device").as_deref() == Some("cuda") { Device::Cuda(0) } else { Device::Cpu };
    synaptix_kernels_cpu::ensure_registered();
    if device.is_cuda() {
        synaptix_kernels_cuda::ensure_registered();
    }
    let t = Instant::now();
    let p = SileroPipeline::from_syn(Path::new(&a[1]), device).expect("load");
    eprintln!("загрузка {:.2} с; голоса: {}", t.elapsed().as_secs_f64(), p.speakers().join(", "));
    if let Some(lines) = get("--lines") {
        let texts = std::fs::read_to_string(lines).unwrap();
        let gold = std::fs::read_to_string(get("--golden").expect("--golden")).unwrap();
        let (mut ok, mut n) = (0, 0);
        let t = Instant::now();
        for (x, g) in texts.lines().zip(gold.lines()) {
            n += 1;
            // эталон — accentor(текст) напрямую (без prepare), так и сравниваем
            let ours = p.accentuate_raw(x).unwrap();
            if ours == g {
                ok += 1;
            } else if n - ok <= 15 {
                eprintln!("  РАЗНОЕ: {x}\n    наш:    {ours}\n    эталон: {g}");
            }
        }
        eprintln!("ударения: {ok}/{n} совпали ({:.1} мс на строку)", t.elapsed().as_secs_f64() * 1e3 / n.max(1) as f64);
        return;
    }
    let text = get("--text").expect("--text");
    let o = Options {
        speaker: get("--speaker").unwrap_or_else(|| p.speakers()[0].clone()),
        sample_rate: get("--sr").map(|s| s.parse().unwrap()).unwrap_or(48_000),
        accents: !a.iter().any(|x| x == "--no-accent"),
        ..Options::default()
    };
    if let Some(r) = get("--refdir").map(PathBuf::from) {
        let (seq, extra) = p.inputs(&text, &o).unwrap();
        let acc = std::fs::read_to_string(r.join("accented.txt")).unwrap_or_default();
        let ours_acc = p.accentuate(&text, None, o.accents).unwrap();
        if !acc.trim().is_empty() {
            eprintln!("ударения: {}", if ours_acc == acc.trim_end_matches('\n') { "совпали".to_string() } else { format!("РАЗНЫЕ\n  наш:    {ours_acc}\n  эталон: {}", acc.trim()) });
        }
        let rseq = i64s(&r.join("seq.npy"));
        eprintln!("seq: {} (эталон {}), {}", seq.len(), rseq.len(), if seq == rseq { "совпали" } else { "РАЗНЫЕ" });
        if r.join("type_ids.npy").exists() {
            let rt = i64s(&r.join("type_ids.npy"));
            eprintln!("type_ids: {}", if extra.type_ids.as_deref() == Some(&rt[..]) { "совпали".to_string() } else { format!("РАЗНЫЕ {:?} vs {rt:?}", extra.type_ids) });
        }
        let t = Instant::now();
        let out = p.infer(&text, &o).unwrap();
        eprintln!("сеть: {:.3} с", t.elapsed().as_secs_f64());
        let rd = f32s(&r.join("dur.npy"));
        let dd = out.dur.iter().zip(&rd).filter(|(a, b)| a != b).count();
        eprintln!("длительности: {} токенов, расходятся {dd} (кадров {} vs {})", out.dur.len(), out.dur.iter().sum::<f32>(), rd.iter().sum::<f32>());
        eprintln!("высота тона: отн. ошибка {:.2e}", rel(&out.pitch, &f32s(&r.join("pitch.npy"))));
        let rm = f32s(&r.join("mel.npy"));
        eprintln!("мел: {} vs {} значений, отн. ошибка {:.2e}", out.mel.len(), rm.len(), rel(&out.mel, &rm));
        let ra = f32s(&r.join("audio.npy"));
        eprintln!("звук: {} vs {} отсчётов, отн. ошибка {:.2e}", out.audio.len(), ra.len(), rel(&out.audio, &ra));
        let v = p.vocode(&rm, 192, o.sample_rate).unwrap();
        eprintln!("вокодер (мел эталона): отн. ошибка {:.2e}", rel(&v, &ra));
    }
    let t = Instant::now();
    let pcm = p.synthesize(&text, &o).unwrap();
    let el = t.elapsed().as_secs_f64();
    let tm = p.last_timings();
    eprintln!("синтез: {:.2} с звука за {el:.3} с (RTF {:.4}; фронтенд {:.3} с, сеть {:.3} с, кусков {})", tm.audio_s, el / tm.audio_s.max(1e-9), tm.frontend_s, tm.net_s, tm.chunks);
    if let Some(f) = get("--out") {
        write_wav(&f, &pcm, o.sample_rate);
        eprintln!("→ {f}");
    }
}

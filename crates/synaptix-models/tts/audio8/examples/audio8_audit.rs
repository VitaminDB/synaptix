//! Сверка с эталоном `reference/audio8_ref.py` и замер скорости.
//!
//! cargo run --profile fast-release -p synaptix-tts-audio8 --example audio8_audit -- \
//!     <bundle.syn> --text "..." [--ref DIR] [--device cuda|cpu] [--seed 0] [--greedy]
//!     [--max-frames N] [--voice a.wav --voice-text "..."] [--out out.wav]
//!     [--decode codes.npy] [--encode a.wav] [--quant mxfp8|nvfp4] [--repeat N]
//!
//! С `--ref DIR` сравнивает с файлами эталона: prompt.npy, first_logits.npy,
//! first_fast.npy, codes.npy, audio.wav (декод тех же кодов нашим кодеком), ref_codes.npy.

use std::path::{Path, PathBuf};
use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_audio8::{Audio8Pipeline, GenerateOptions};

fn npy(path: &Path) -> (Vec<usize>, String, Vec<u8>) {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&b[..6], b"\x93NUMPY");
    let (hl, off) = if b[6] == 1 { (u16::from_le_bytes([b[8], b[9]]) as usize, 10) } else { (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12) };
    let h = String::from_utf8_lossy(&b[off..off + hl]).to_string();
    let descr = h.split("'descr':").nth(1).unwrap().split('\'').nth(1).unwrap().to_string();
    let shape_s = h.split("'shape':").nth(1).unwrap();
    let shape_s = &shape_s[shape_s.find('(').unwrap() + 1..shape_s.find(')').unwrap()];
    let shape: Vec<usize> = shape_s.split(',').filter_map(|s| s.trim().parse().ok()).collect();
    (shape, descr, b[off + hl..].to_vec())
}

fn npy_i64(path: &Path) -> (Vec<usize>, Vec<i64>) {
    let (s, d, b) = npy(path);
    assert_eq!(d, "<i8", "{}", path.display());
    (s, b.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
}

fn npy_f32(path: &Path) -> Vec<f32> {
    let (_, d, b) = npy(path);
    assert_eq!(d, "<f4", "{}", path.display());
    b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn rows_from(shape: &[usize], v: &[i64]) -> Vec<Vec<u32>> {
    (0..shape[0]).map(|r| v[r * shape[1]..(r + 1) * shape[1]].iter().map(|&x| x as u32).collect()).collect()
}

fn rel(a: &[f32], b: &[f32]) -> (f32, f32) {
    let n = a.len().min(b.len());
    let (mut d2, mut b2, mut mx) = (0f64, 0f64, 0f32);
    for i in 0..n {
        let d = a[i] - b[i];
        d2 += (d as f64) * (d as f64);
        b2 += (b[i] as f64) * (b[i] as f64);
        mx = mx.max(d.abs());
    }
    (((d2 / b2.max(1e-30)).sqrt()) as f32, mx)
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold(0, |b, (i, &x)| if x > v[b] { i } else { b })
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let get = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let has = |k: &str| a.iter().any(|x| x == k);
    let bundle = PathBuf::from(&a[1]);
    let device = match get("--device").as_deref() {
        Some("cpu") => Device::Cpu,
        _ => Device::Cuda(0),
    };
    synaptix_kernels_cpu::ensure_registered();
    if device.is_cuda() {
        synaptix_kernels_cuda::ensure_registered();
    }
    let quant = get("--quant").map(|q| match q.as_str() {
        "mxfp8" => DType::MXFP8,
        "nvfp4" => DType::NVFP4,
        o => panic!("--quant {o}"),
    });
    let t = Instant::now();
    let compute = if device.is_cuda() { DType::BF16 } else { DType::F32 };
    let p = Audio8Pipeline::from_syn_quant(&bundle, device, compute, quant).expect("load");
    eprintln!("загрузка {:.2} с", t.elapsed().as_secs_f64());
    let refdir = get("--ref").map(PathBuf::from);

    if let Some(c) = get("--decode") {
        let (s, v) = npy_i64(Path::new(&c));
        let codes = rows_from(&s, &v);
        let t = Instant::now();
        let pcm = p.codec.decode(&codes).expect("decode");
        eprintln!("декод {} кадров: {:.3} с", s[1], t.elapsed().as_secs_f64());
        if let Some(r) = &refdir {
            let (refpcm, _) = synaptix_audio::io::read_wav_mono_f32(r.join("audio.wav")).expect("ref wav");
            let (e, mx) = rel(&pcm, &refpcm);
            eprintln!("кодек: {} vs {} семплов, отн. ошибка {e:.2e}, макс {mx:.2e}", pcm.len(), refpcm.len());
        }
        if let Some(o) = get("--out") {
            synaptix_audio::io::write_wav_mono_f32(o, &pcm, 44100).unwrap();
        }
        return;
    }
    if let Some(w) = get("--encode") {
        let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(&w).expect("wav");
        let t = Instant::now();
        let vp = p.voice_prompt(&pcm, sr, "x").expect("encode");
        eprintln!("энкод {:.2} с звука → {} кадров: {:.3} с", pcm.len() as f32 / sr as f32, vp.frames(), t.elapsed().as_secs_f64());
        if let Some(r) = &refdir {
            let (s, v) = npy_i64(&r.join("ref_codes.npy"));
            let rc = rows_from(&s, &v);
            for (i, (a, b)) in vp.codes.iter().zip(&rc).enumerate() {
                let same = a.iter().zip(b).filter(|(x, y)| x == y).count();
                eprintln!("  кодбук {i}: совпало {same}/{}", b.len());
            }
        }
        return;
    }

    let text = get("--text").expect("--text");
    let mut o = GenerateOptions { seed: get("--seed").map(|s| s.parse().unwrap()).unwrap_or(0), greedy: has("--greedy"), ..p.default_options() };
    if let Some(m) = get("--max-frames") {
        o.max_frames = m.parse().unwrap();
    }
    let voice = get("--voice").map(|w| {
        let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(&w).expect("voice wav");
        let t = Instant::now();
        let v = p.voice_prompt(&pcm, sr, &get("--voice-text").expect("--voice-text")).expect("voice");
        eprintln!("образец: {} кадров, {:.3} с", v.frames(), t.elapsed().as_secs_f64());
        v
    });

    if let Some(r) = &refdir {
        let rows = p.prompt_rows(&text, voice.as_ref()).unwrap();
        let (s, v) = npy_i64(&r.join("prompt.npy"));
        let rr = rows_from(&s, &v);
        let same_tok = rows[0] == rr[0];
        let same_all = rows == rr;
        eprintln!("промпт: {} токенов (эталон {}), токены {}, коды {}", rows[0].len(), s[1], if same_tok { "совпали" } else { "РАЗНЫЕ" }, if same_all { "совпали" } else { "разные" });
        if !same_tok {
            eprintln!("  наш:    {:?}", &rows[0]);
            eprintln!("  эталон: {:?}", &rr[0]);
        }
        // Первый медленный шаг на промпте эталона.
        let mut kv = p.model.slow.new_cache(p.config.max_seq_len).unwrap();
        let (_, fin) = p.model.slow_step(&rr, &mut kv, 0).unwrap();
        let full = p.model.full_logits(&fin).unwrap();
        let rl = npy_f32(&r.join("first_logits.npy"));
        let (e, mx) = rel(&full, &rl);
        eprintln!("логиты 1-го шага: отн. {e:.2e}, макс {mx:.2e}, argmax {} / {}", argmax(&full), argmax(&rl));
        let rf = r.join("first_fast.npy");
        if rf.exists() {
            let codes = npy_i64(&r.join("codes.npy"));
            let sem = codes.1[0] as u32;
            let mut fkv = p.model.fast.new_cache(p.config.num_codebooks).unwrap();
            p.model.fast_prime(&fin, &mut fkv).unwrap();
            let x = p.model.fast_embed(sem).unwrap();
            let fl = p.model.fast_step(&x, &mut fkv, 1).unwrap();
            let (e, mx) = rel(&fl, &npy_f32(&rf));
            eprintln!("логиты кодбука 1: отн. {e:.2e}, макс {mx:.2e}");
        }
    }

    let reps: usize = get("--repeat").map(|s| s.parse().unwrap()).unwrap_or(1);
    let mut pcm = Vec::new();
    let mut codes = Vec::new();
    for _ in 0..reps {
        let t = Instant::now();
        codes = p.generate_codes(&text, voice.as_ref(), &o).expect("generate");
        let tg = t.elapsed().as_secs_f64();
        let t2 = Instant::now();
        pcm = p.codec.decode(&codes).expect("decode");
        let tc = t2.elapsed().as_secs_f64();
        let tm = p.last_timings();
        let dur = pcm.len() as f64 / 44100.0;
        eprintln!(
            "промпт {} ток., кадров {} (EOS {}), звук {dur:.2} с; префилл {:.3} с, кадры {:.2} с ({:.1} кадр/с), кодек {tc:.3} с; RTF {:.3}",
            tm.prompt_tokens,
            tm.frames,
            tm.finished,
            tm.prefill_s,
            tm.decode_s,
            tm.frames as f64 / tm.decode_s.max(1e-9),
            (tg + tc) / dur.max(1e-9)
        );
    }
    if let Some(r) = &refdir {
        let (s, v) = npy_i64(&r.join("codes.npy"));
        let rc = rows_from(&s, &v);
        let n = rc[0].len().min(codes[0].len());
        let first_diff = (0..n).find(|&t| (0..rc.len()).any(|i| rc[i][t] != codes[i][t]));
        eprintln!("коды: наши {} кадров, эталон {}; первое расхождение: {:?}", codes[0].len(), rc[0].len(), first_diff);
        if let Some(f) = first_diff {
            for t in f.saturating_sub(1)..(f + 2).min(n) {
                eprintln!("  кадр {t}: наш {:?}", (0..rc.len()).map(|i| codes[i][t]).collect::<Vec<_>>());
                eprintln!("  кадр {t}: эт. {:?}", (0..rc.len()).map(|i| rc[i][t]).collect::<Vec<_>>());
            }
        }
    }
    if let Some(out) = get("--out") {
        synaptix_audio::io::write_wav_mono_f32(&out, &pcm, 44100).unwrap();
        eprintln!("→ {out}");
    }
}

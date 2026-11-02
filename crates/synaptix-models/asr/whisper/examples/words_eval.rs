//! Проверка слов Whisper: WER по эталонному тексту и ошибка границ слов против
//! принудительного выравнивания MFA (synaptix-align-mfa) по эталонному тексту.
//!   words_eval <модель.syn> <список.tsv> --lang en --mfa <model.mfa> [--verbatim] [--plain] [--show] [--cpu]
//! Список: `путь.wav<TAB>эталонный текст` на строку. `--plain` — ещё и время
//! обычного transcribe для сравнения скорости.
use synaptix_align_mfa::{align_alts, AlignOptions, AlignWord, AlignWordAlts, MfaModel};
use synaptix_asr_whisper::{Task, WhisperPipeline, WordOptions};
use synaptix_core::{device::Device, dtype::DType};

/// Нижний регистр, без пунктуации (апостроф внутри слова сохраняется).
fn norm_words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || *c == '\'')
                .collect::<String>()
                .trim_matches('\'')
                .to_string()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

/// Выравнивание Левенштейна: пары (i_ref, i_hyp) совпавших слов и число ошибок.
fn lev_align(r: &[String], h: &[String]) -> (Vec<(usize, usize)>, usize) {
    let (n, m) = (r.len(), h.len());
    let mut d = vec![vec![0usize; m + 1]; n + 1];
    for i in 0..=n {
        d[i][0] = i;
    }
    for j in 0..=m {
        d[0][j] = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let s = d[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1]);
            d[i][j] = s.min(d[i - 1][j] + 1).min(d[i][j - 1] + 1);
        }
    }
    let (mut i, mut j, mut pairs) = (n, m, Vec::new());
    while i > 0 && j > 0 {
        let s = d[i - 1][j - 1] + usize::from(r[i - 1] != h[j - 1]);
        if d[i][j] == s {
            if r[i - 1] == h[j - 1] {
                pairs.push((i - 1, j - 1));
            }
            i -= 1;
            j -= 1;
        } else if d[i][j] == d[i - 1][j] + 1 {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    pairs.reverse();
    (pairs, d[n][m])
}

fn mfa_words(model: &MfaModel, pcm: &[f32], words: &[String]) -> Option<Vec<(f32, f32)>> {
    let alts: Option<Vec<AlignWordAlts>> = words
        .iter()
        .map(|w| model.lexicon.get(w).map(|ps| AlignWordAlts { alts: ps.iter().map(|p| (AlignWord::from_pron(p), p.prob)).collect() }))
        .collect();
    let feats = model.features(pcm);
    let al = align_alts(model, &feats, &alts?, &AlignOptions::default()).ok()?;
    Some(al.words.iter().map(|w| (w.start as f32 * al.frame_shift, w.end as f32 * al.frame_shift)).collect())
}

fn stats(v: &mut [f32]) -> String {
    if v.is_empty() {
        return "—".into();
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let within = |t: f32| 100.0 * v.iter().filter(|&&x| x <= t).count() as f32 / v.len() as f32;
    format!(
        "среднее {:.0} мс, медиана {:.0} мс, p90 {:.0} мс, ≤100 мс {:.0}%, ≤200 мс {:.0}%",
        mean * 1e3,
        v[v.len() / 2] * 1e3,
        v[v.len() * 9 / 10] * 1e3,
        within(0.1),
        within(0.2)
    )
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let a: Vec<String> = std::env::args().collect();
    let arg = |k: &str| a.iter().position(|x| x == k).and_then(|i| a.get(i + 1)).cloned();
    let flag = |k: &str| a.iter().any(|x| x == k);
    let lang = arg("--lang");
    let (dev, dt) = if flag("--cpu") { (Device::Cpu, DType::F32) } else { (Device::Cuda(0), DType::F16) };
    let p = WhisperPipeline::from_syn(&a[1], dev, dt).expect("модель");
    let mfa = arg("--mfa").map(|m| MfaModel::load(std::path::Path::new(&m)).expect("mfa"));
    let opts = WordOptions { verbatim: flag("--verbatim"), initial_prompt: arg("--prompt"), snap_to_pauses: !flag("--no-snap") };
    let list = std::fs::read_to_string(&a[2]).expect("список");

    let (mut errs, mut nref) = (0usize, 0usize);
    let (mut d_start, mut d_end, mut d_pause) = (Vec::new(), Vec::new(), Vec::new());
    let (mut bias_s, mut bias_e) = (Vec::new(), Vec::new());
    let (mut t_words, mut t_plain, mut dur) = (0f64, 0f64, 0f64);
    let mut probs_ok = Vec::new();
    let mut probs_bad = Vec::new();
    // прогрев (граф декода, ядра)
    let warm = vec![0f32; 16000];
    let _ = p.transcribe_words(&warm, lang.as_deref(), &opts);
    for line in list.lines().filter(|l| !l.trim().is_empty()) {
        let mut cols = line.split('\t');
        let (path, text) = (cols.next().unwrap(), cols.next().expect("путь<TAB>текст"));
        let audio = WhisperPipeline::load_audio(path).expect("аудио");
        dur += audio.len() as f64 / 16000.0;
        if flag("--plain") {
            let t = std::time::Instant::now();
            let _ = p.transcribe(&audio, lang.as_deref(), Task::Transcribe).expect("transcribe");
            t_plain += t.elapsed().as_secs_f64();
        }
        let t = std::time::Instant::now();
        // GPU делят другие процессы: при нехватке памяти — ждём и повторяем.
        let mut tries = 0;
        let r = loop {
            match p.transcribe_words(&audio, lang.as_deref(), &opts) {
                Ok(r) => break r,
                Err(e) if tries < 60 => {
                    tries += 1;
                    eprintln!("повтор ({e})");
                    std::thread::sleep(std::time::Duration::from_secs(10));
                }
                Err(e) => panic!("распознавание: {e}"),
            }
        };
        t_words += t.elapsed().as_secs_f64();

        let rw = norm_words(text);
        // слово гипотезы → нормализованные части (одно слово Whisper может дать 0..n)
        let mut hw = Vec::new();
        let mut hidx = Vec::new();
        for (i, w) in r.words.iter().enumerate() {
            for nw in norm_words(&w.text) {
                hw.push(nw);
                hidx.push(i);
            }
        }
        let (pairs, e) = lev_align(&rw, &hw);
        errs += e;
        nref += rw.len();
        let matched: std::collections::HashSet<usize> = pairs.iter().map(|&(_, j)| hidx[j]).collect();
        for (i, w) in r.words.iter().enumerate() {
            if matched.contains(&i) { probs_ok.push(w.prob) } else { probs_bad.push(w.prob) }
        }
        let name = std::path::Path::new(path).file_name().unwrap().to_string_lossy().to_string();
        let mut line_err = String::new();
        if let Some(m) = &mfa {
            // третья колонка — список частей (wav<TAB>текст), склеенных с паузой
            // 0.7 с: эталон — выравнивание каждой части со смещением
            let parts = line.split('\t').nth(2);
            let refw = match parts {
                None => mfa_words(m, &audio, &rw),
                Some(pl) => {
                    let mut all = Vec::new();
                    let mut off = 0f32;
                    let mut ok = true;
                    for l in std::fs::read_to_string(pl).expect("части").lines().filter(|l| !l.is_empty()) {
                        let (pp, pt) = l.split_once('\t').unwrap();
                        let pa = WhisperPipeline::load_audio(pp).unwrap();
                        match mfa_words(m, &pa, &norm_words(pt)) {
                            Some(v) => all.extend(v.into_iter().map(|(a, b)| (a + off, b + off))),
                            None => ok = false,
                        }
                        off += pa.len() as f32 / 16000.0 + 0.7;
                    }
                    (ok && all.len() == rw.len()).then_some(all)
                }
            };
            match refw {
                Some(refw) => {
                    let mut ds = Vec::new();
                    for &(ri, hj) in &pairs {
                        let w = &r.words[hidx[hj]];
                        let (s, e) = refw[ri];
                        ds.push((w.start - s).abs());
                        d_start.push((w.start - s).abs());
                        d_end.push((w.end - e).abs());
                        bias_s.push(w.start - s);
                        bias_e.push(w.end - e);
                        // паузы эталона ≥ 150 мс: сравниваем длину паузы
                        if ri + 1 < rw.len() {
                            let gap = refw[ri + 1].0 - e;
                            if gap >= 0.15 {
                                if let Some(&(_, hj2)) = pairs.iter().find(|&&(r2, _)| r2 == ri + 1) {
                                    let hgap = r.words[hidx[hj2]].start - w.end;
                                    d_pause.push((hgap - gap).abs());
                                }
                            }
                        }
                    }
                    let m = ds.iter().sum::<f32>() / ds.len().max(1) as f32;
                    line_err = format!("ошибка начала {:.0} мс", m * 1e3);
                    if flag("--show") {
                        for &(ri, hj) in &pairs {
                            let w = &r.words[hidx[hj]];
                            println!("    {:>14} whisper {:6.2}-{:6.2}  mfa {:6.2}-{:6.2}", rw[ri], w.start, w.end, refw[ri].0, refw[ri].1);
                        }
                    }
                }
                None => line_err = "MFA: нет слова в словаре/не выровнялось".into(),
            }
        }
        println!("{name}: ошибок {e}/{} {line_err}\n  REF: {}\n  HYP: {}", rw.len(), text, r.text);
    }
    println!("\n== итог: {} с аудио", dur.round());
    println!("WER {:.2}% ({errs}/{nref})", 100.0 * errs as f64 / nref.max(1) as f64);
    println!("начало слова: {}", stats(&mut d_start));
    println!("конец слова:  {}", stats(&mut d_end));
    let avg = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
    println!("смещение (whisper − mfa): начало {:+.0} мс, конец {:+.0} мс", avg(&bias_s) * 1e3, avg(&bias_e) * 1e3);
    println!("длина паузы ≥150 мс: {} (n={})", stats(&mut d_pause), d_pause.len());
    println!("уверенность: верные слова {:.3} (n={}), ошибочные {:.3} (n={})", avg(&probs_ok), probs_ok.len(), avg(&probs_bad), probs_bad.len());
    println!("скорость: со словами {:.3} с на минуту аудио{}", t_words / dur * 60.0,
        if flag("--plain") { format!(", обычный transcribe {:.3} с на минуту", t_plain / dur * 60.0) } else { String::new() });
}

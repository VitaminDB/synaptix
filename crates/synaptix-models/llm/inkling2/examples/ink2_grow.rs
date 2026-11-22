//! Рост одной сессии по ходам, как в службе: режим памяти следует за длиной.
//!
//!   ink2_grow <модель> <корпус.txt> [--stages 8192,60000,200000] [--gen 128]
//!             [--window 1048576] [--kl 2048] [--ref auto|bf16|mxfp8] [--needle-gen 400]
//!
//! Модель грузится как в службе (предел — окно 1M). Промпт — «иголка» и корпус;
//! ход k — первые `stages[k]` токенов (кратно 512), сессия дорастает правилом
//! службы (шаг 16K), ход берёт прошлый промпт из кэша через точку возврата, затем
//! `--gen` токенов жадно. На каждом участке печатаются режим экспертов, слоты,
//! формат KV, скорость префилла и декода. На последнем — вопрос об иголке.
//! `--kl N`: логиты N следующих токенов корпуса (teacher forcing) против эталона
//! без переходов — новой сессии той же ёмкости в конечном режиме (`--ref`:
//! формат KV эталона; auto — как у сессии после переходов), префилл с нуля той же
//! нарезкой: KL, топ-1, max|Δ|, строки бит в бит. `--ref-chunk N` — эталон другой нарезкой
//! префилла (порог шума: только порядок суммирования).
//! `--cap N` — первая сессия сразу на N позиций (проверка раскладки памяти на окне целиком).
//! `--swap N` — сценарий службы с двумя сессиями: большая уезжает в RAM, новая
//! короткая на N токенов (режим должен подняться — эксперты снова на карту),
//! она уезжает в RAM, большая возвращается (режим ужимается) и продолжает
//! из кэша. У каждого хода печатается отпечаток токенов (детерминизм).

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::generate::GenerationConfig;
use synaptix_llm_inkling2::{Inkling2Pipeline, Inkling2Session, KvFormat};

const NEEDLE: &str = "Важная заметка для проверки памяти: кодовое слово хранителя архива — «СИНИЙ-ВЕРБЛЮД-4718». Запомни его.\n\n";
const QUESTION: &str = "\n\nКонец материалов. Вопрос: какое кодовое слово хранителя архива было названо в самом начале? Ответь одной строкой.";
const SESSION_STEP: usize = 16384;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn gpu_used_mb() -> String {
    std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

fn cfg(max_new: usize) -> GenerationConfig {
    GenerationConfig {
        max_new_tokens: max_new,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        seed: 7,
        eos_token_id: None,
        eos_token_ids: Vec::new(),
        max_seq: None,
        prefill_batch: 512,
        constraint: None,
    }
}

fn state(p: &Inkling2Pipeline, s: &Inkling2Session) -> String {
    let mode = match p.hub.mode() {
        Some(synaptix_llm_inkling2::ExpertMode::Resident) => "эксперты на карте".to_string(),
        Some(synaptix_llm_inkling2::ExpertMode::Zc) => format!("zero-copy {} слотов", p.hub.slots()),
        None => "эксперты не подняты".into(),
    };
    format!("{mode}, KV {}, ёмкость {}, VRAM {} МБ", s.kv_format().name(), s.ctx_tokens(), gpu_used_mb())
}

/// Логиты строк хвоста `ids` после кэша сессии (на хост, по строкам).
fn rows(p: &Inkling2Pipeline, s: &mut Inkling2Session, ids: &[u32]) -> Result<(Vec<Vec<f32>>, usize), Box<dyn std::error::Error>> {
    let mut out = Vec::new();
    let reuse = p.prefill_cached_rows(s, ids, &mut |t: &Tensor| {
        let (n, v) = (t.dims()[0], t.dims()[1]);
        let h: Vec<f32> = t.to_device(Device::Cpu).and_then(|x| x.flatten_all()).and_then(|x| x.to_vec1::<f32>()).map_err(|e| synaptix_llm_inkling2::PipelineError::Load(e.to_string()))?;
        for r in 0..n {
            out.push(h[r * v..(r + 1) * v].to_vec());
        }
        Ok(())
    })?;
    Ok((out, reuse))
}

fn fp(t: &[u32]) -> u64 {
    t.iter().fold(0xcbf29ce484222325u64, |h, &x| (h ^ x as u64).wrapping_mul(0x100000001b3))
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
    let l = m + s.ln();
    v.iter().map(|&x| x as f64 - l).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let model = &args[1];
    let corpus = std::fs::read_to_string(&args[2])?;
    let stages: Vec<usize> = arg(&args, "--stages")
        .unwrap_or_else(|| "8192,60000,200000".into())
        .split(',')
        .map(|v| v.trim().parse::<usize>().map(|n| n.div_ceil(512) * 512))
        .collect::<Result<_, _>>()?;
    let gen: usize = arg(&args, "--gen").and_then(|v| v.parse().ok()).unwrap_or(128);
    let window: usize = arg(&args, "--window").and_then(|v| v.parse().ok()).unwrap_or(1 << 20);
    let kl: usize = arg(&args, "--kl").and_then(|v| v.parse().ok()).unwrap_or(0);
    let needle_gen: usize = arg(&args, "--needle-gen").and_then(|v| v.parse().ok()).unwrap_or(0);
    let ref_fmt = arg(&args, "--ref").unwrap_or_else(|| "auto".into());
    let swap: usize = arg(&args, "--swap").and_then(|v| v.parse().ok()).unwrap_or(0);
    let cap0: usize = arg(&args, "--cap").and_then(|v| v.parse().ok()).unwrap_or(0);
    let dev = Device::Cuda(0);

    let t0 = Instant::now();
    let p = Inkling2Pipeline::load_with_ctx(model, dev, Some(window))?;
    eprintln!("загрузка {:.1} с, VRAM {} МБ", t0.elapsed().as_secs_f64(), gpu_used_mb());

    let mut ids = p.encode(&format!("<|message_system|><|content_text|>Thinking effort level: 0<|end_message|><|message_user|><|content_text|>{NEEDLE}"))?;
    let tail = p.encode(&format!("{QUESTION}<|end_message|>"))?;
    let last = *stages.last().expect("участки");
    let body = p.encode(&corpus)?;
    let want_body = last + kl.max(512) + 16;
    if body.len() + ids.len() < want_body {
        return Err(format!("корпус короче нужного: {} < {want_body}", body.len()).into());
    }
    ids.extend_from_slice(&body[..want_body - ids.len()]);
    eprintln!("корпус {} токенов, участки {stages:?}", body.len());

    let want = |need: usize| need.div_ceil(SESSION_STEP).saturating_mul(SESSION_STEP).clamp(need, window);
    let mut s: Option<Inkling2Session> = None;
    for (k, &st) in stages.iter().enumerate() {
        let need = st + gen + 16;
        let t = Instant::now();
        match s.as_mut() {
            None => s = Some(p.new_session(want(need).max(cap0))?),
            Some(s) if s.ctx_tokens() < need => p.grow_session(s, want(need))?,
            _ => {}
        }
        let s = s.as_mut().expect("сессия");
        let grow_s = t.elapsed().as_secs_f64();
        let (out, stt, reuse) = p.generate_cached_streaming(s, &ids[..st], &cfg(gen), &mut |_: u32| true)?;
        let fresh = st - reuse;
        let dec = if stt.new_tokens > 1 { (stt.new_tokens - 1) as f64 * 1000.0 / stt.decode_ms.max(1) as f64 } else { 0.0 };
        let hits = p.model.zc_stats().map(|(h, n)| format!(", попадания zc {:.1} %", 100.0 * h as f64 / n.max(1) as f64)).unwrap_or_default();
        println!(
            "участок {k}: {st} ток. | рост/создание {grow_s:.2} с | {} | из кэша {reuse}, префилл {fresh} ток. за {:.1} с ({:.0} ток/с) | декод {} ток. {dec:.1} ток/с{hits} | отпечаток {:016x}",
            state(&p, s),
            stt.prefill_ms as f64 / 1000.0,
            fresh as f64 * 1000.0 / stt.prefill_ms.max(1) as f64,
            stt.new_tokens,
            fp(&out)
        );
    }
    let mut s = s.expect("сессия");

    if swap > 0 {
        let t = Instant::now();
        let moved = s.park_to_host()?;
        println!("swap: большая в RAM ({:.1} ГБ за {:.2} с) | VRAM {} МБ", moved as f64 / (1u64 << 30) as f64, t.elapsed().as_secs_f64(), gpu_used_mb());
        let t = Instant::now();
        let mut small = p.new_session(want(swap + gen + 16))?;
        let created = t.elapsed().as_secs_f64();
        let (out, stt, _) = p.generate_cached_streaming(&mut small, &ids[..swap], &cfg(gen), &mut |_: u32| true)?;
        println!(
            "swap: короткая {swap} ток. создана за {created:.2} с | {} | префилл {:.1} с, декод {:.1} ток/с | отпечаток {:016x}",
            state(&p, &small),
            stt.prefill_ms as f64 / 1000.0,
            (stt.new_tokens - 1) as f64 * 1000.0 / stt.decode_ms.max(1) as f64,
            fp(&out)
        );
        small.park_to_host()?;
        let t = Instant::now();
        s.unpark_to(dev)?;
        println!("swap: большая вернулась за {:.2} с | {}", t.elapsed().as_secs_f64(), state(&p, &s));
        let more = last + 512;
        let need = more + gen + 16;
        if s.ctx_tokens() < need {
            p.grow_session(&mut s, want(need))?;
        }
        let (out, stt, reuse) = p.generate_cached_streaming(&mut s, &ids[..more], &cfg(gen), &mut |_: u32| true)?;
        println!(
            "swap: ход большой — из кэша {reuse} из {more}, первый токен через {:.2} с, декод {:.1} ток/с | {} | отпечаток {:016x}",
            stt.prefill_ms as f64 / 1000.0,
            (stt.new_tokens - 1) as f64 * 1000.0 / stt.decode_ms.max(1) as f64,
            state(&p, &s),
            fp(&out)
        );
        drop(small);
    }

    if needle_gen > 0 {
        let mut prompt = ids[..last].to_vec();
        prompt.extend_from_slice(&tail);
        let need = prompt.len() + needle_gen + 16;
        if s.ctx_tokens() < need {
            p.grow_session(&mut s, want(need))?;
        }
        s.set_resume_at(Some(last));
        let mut c = cfg(needle_gen);
        c.eos_token_ids = p.eos_ids();
        let (out, stt, reuse) = p.generate_cached_streaming(&mut s, &prompt, &c, &mut |_: u32| true)?;
        let text = p.decode(&out)?;
        println!(
            "иголка: из кэша {reuse} из {}, первый токен через {:.2} с, {} ток. {:.1} ток/с | {} | верно: {}\n{text}",
            prompt.len(),
            stt.prefill_ms as f64 / 1000.0,
            stt.new_tokens,
            (stt.new_tokens.max(1) - 1) as f64 * 1000.0 / stt.decode_ms.max(1) as f64,
            state(&p, &s),
            text.contains("СИНИЙ-ВЕРБЛЮД-4718")
        );
    }

    if kl > 0 {
        let full = &ids[..last + kl];
        let t = Instant::now();
        let (a, reuse) = rows(&p, &mut s, full)?;
        println!("KL: после переходов {} строк, из кэша {reuse} ({:.1} с) | {}", a.len(), t.elapsed().as_secs_f64(), state(&p, &s));
        let cap = s.ctx_tokens();
        let fmt = match ref_fmt.as_str() {
            "auto" => s.kv_format(),
            f => KvFormat::parse(f).ok_or("--ref: формат")?,
        };
        drop(s);
        // `--ref-chunk N` — эталон другой нарезкой префилла: только порядок суммирования
        // (части flash-внимания), математика та же — порог шума реализации.
        if let Some(c) = arg(&args, "--ref-chunk") {
            // SAFETY: однопоточный пример, переменную читает только этот поток.
            unsafe { std::env::set_var("SYN_INKLING2_PREFILL_CHUNK", c) };
        }
        let t = Instant::now();
        let mut r = p.new_session_fmt(cap, Some(fmt))?;
        p.prefill_cached(&mut r, &ids[..last])?;
        let (b, reuse_b) = rows(&p, &mut r, full)?;
        println!("эталон: {} с нуля за {:.1} с, из кэша {reuse_b} | {}", ref_fmt, t.elapsed().as_secs_f64(), state(&p, &r));
        let (mut klsum, mut top1, mut same, mut maxd) = (0f64, 0usize, 0usize, 0f32);
        let mut kls = Vec::with_capacity(a.len());
        for (x, y) in a.iter().zip(&b) {
            let (lx, ly) = (log_softmax(x), log_softmax(y));
            let k: f64 = ly.iter().zip(&lx).map(|(&q, &pp)| q.exp() * (q - pp)).sum();
            klsum += k;
            kls.push(k);
            let am = |v: &[f32]| v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i);
            top1 += (am(x) == am(y)) as usize;
            same += (x == y) as usize;
            maxd = x.iter().zip(y).map(|(p, q)| (p - q).abs()).fold(maxd, f32::max);
        }
        kls.sort_by(|a, b| a.total_cmp(b));
        let n = a.len().max(1);
        println!(
            "KL к эталону: {:.6} (p99 {:.5}), топ-1 {:.2} %, строк бит в бит {same}/{n}, max|Δ| {maxd:.4}",
            klsum / n as f64,
            kls[(kls.len() * 99 / 100).min(kls.len() - 1)],
            100.0 * top1 as f64 / n as f64
        );
    }
    Ok(())
}

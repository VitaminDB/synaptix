//! Точка возврата сессии Inkling-2: ход 2 с промптом «промпт 1 + ответ 1 без
//! размышлений + новое сообщение» после восстановления снимка против хода с
//! нуля на том же промпте.
//!
//!   ink2_resume <модель> <корпус.txt> [--len 8000] [--gen 48] [--history]
//!
//! `--history` — точка возврата на конце истории (как зовёт служба:
//! `set_resume_at(history)`), иначе — на конце промпта. Сверка:
//! - логиты последней позиции промпта 2: восстановление против хода с нуля с
//!   той же нарезкой префилла (граница чанка на точке) — бит в бит;
//! - то же против хода с нуля с обычной нарезкой (чанки по 512 через точку) —
//!   отличие на уровне шума, печатаются max|Δ|, KL, топ-1;
//! - токены декода после восстановления (граф декода прошлого хода) против
//!   хода с нуля — совпадают целиком.

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::generate::GenerationConfig;
use synaptix_llm_inkling2::Inkling2Pipeline;

fn arg(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let s: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
    v.iter().map(|&x| x as f64 - m - s.ln()).collect()
}

/// (max|Δ|, KL(a‖b), топ-1 совпал, бит в бит).
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, bool, bool) {
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
    let (la, lb) = (log_softmax(a), log_softmax(b));
    let kl: f64 = la.iter().zip(&lb).map(|(x, y)| x.exp() * (x - y)).sum();
    let top = |v: &[f32]| v.iter().enumerate().max_by(|p, q| p.1.total_cmp(q.1)).unwrap().0;
    let bits = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    (max, kl, top(a) == top(b), bits)
}

fn cfg(n: usize) -> GenerationConfig {
    GenerationConfig { max_new_tokens: n, temperature: 0.0, top_k: 0, top_p: 1.0, ..Default::default() }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let args: Vec<String> = std::env::args().collect();
    let corpus = std::fs::read_to_string(&args[2])?;
    let len: usize = arg(&args, "--len").and_then(|v| v.parse().ok()).unwrap_or(8000);
    let gen: usize = arg(&args, "--gen").and_then(|v| v.parse().ok()).unwrap_or(48);
    let at_history = args.iter().any(|a| a == "--history");
    let ctx = len + 2 * gen + 512;
    let p = Inkling2Pipeline::load_with_ctx(&args[1], Device::Cuda(0), Some(ctx))?;
    let enc = |s: &str| p.encode(s).unwrap();

    // Промпт 1: история (реплика пользователя) + приглашение к ответу.
    let mut text = enc(&corpus);
    text.truncate(len);
    let mut history1 = enc("<|message_user|><|content_text|>Прочитай текст и кратко перескажи его.\n\n");
    history1.extend(&text);
    history1.extend(enc("<|end_message|>"));
    let invite = enc("<|message_model|>");
    let prompt1: Vec<u32> = history1.iter().chain(&invite).copied().collect();
    let point = if at_history { history1.len() } else { prompt1.len() };
    println!("промпт 1: {} ток., точка возврата на {point} ({})", prompt1.len(), if at_history { "конец истории" } else { "конец промпта" });

    // Ход 1 в сессии: с размышлениями (что сгенерирует — то и будет), декод через граф.
    let mut s = p.new_session(ctx)?;
    s.set_resume_at(at_history.then_some(point));
    let t = Instant::now();
    let (out1, st1, r1) = p.generate_cached_streaming(&mut s, &prompt1, &cfg(gen), &mut |_: u32| true)?;
    println!(
        "ход 1: из кэша {r1}, префилл {} мс, {} ток. за {} мс ({:.1} ток/с), всего {} мс; снимок {:?}",
        st1.prefill_ms,
        st1.new_tokens,
        st1.decode_ms,
        st1.new_tokens as f64 * 1000.0 / st1.decode_ms.max(1) as f64,
        t.elapsed().as_millis(),
        s.snapshot_info()
    );
    println!("ответ 1: {:?}", p.decode(&out1).unwrap_or_default());

    // Промпт 2: история + ответ 1 без размышлений + новое сообщение.
    let mut prompt2 = history1.clone();
    prompt2.extend(enc("<|message_model|><|content_text|>Текст о разном; пересказ ниже.<|end_message|>"));
    prompt2.extend(enc("<|message_user|><|content_text|>Назови одним словом главную тему текста.<|end_message|><|message_model|>"));
    if !at_history {
        // Без границы службы точка на конце промпта 1 — промпт 2 обязан его продолжать.
        prompt2 = prompt1.iter().chain(&prompt2[history1.len()..]).copied().collect();
    }
    println!("промпт 2: {} ток., reusable = {}", prompt2.len(), s.reusable(&prompt2));
    assert_eq!(s.reusable(&prompt2), point, "из кэша — весь промпт до точки");

    // A: восстановление снимка. Граница службы — та же точка: она уже стоит и остаётся
    // (иначе ход снял бы новую на конце промпта 2).
    s.set_resume_at(Some(point));
    let t = Instant::now();
    let (la, ra) = p.prefill_cached(&mut s, &prompt2)?;
    let la = host(&la);
    let ta = t.elapsed().as_millis();
    // B: с нуля, граница чанка на той же точке.
    let mut f = p.new_session(ctx)?;
    f.set_resume_at(Some(point));
    let t = Instant::now();
    let (lb, rb) = p.prefill_cached(&mut f, &prompt2)?;
    let lb = host(&lb);
    let tb = t.elapsed().as_millis();
    drop(f);
    // C: с нуля, обычная нарезка.
    let mut g = p.new_session(ctx)?;
    let (lc, _) = p.prefill_cached(&mut g, &prompt2)?;
    let lc = host(&lc);
    drop(g);
    println!("ход 2: из кэша {ra} ({ta} мс) против с нуля {rb} ({tb} мс)");
    let (m, kl, top, bits) = compare(&la, &lb);
    println!("восстановление ↔ с нуля (та же нарезка): бит в бит {bits}, max|Δ| {m:e}, KL {kl:e}, топ-1 {top}");
    let (m2, kl2, top2, bits2) = compare(&la, &lc);
    println!("восстановление ↔ с нуля (чанки 512 через точку): бит в бит {bits2}, max|Δ| {m2:e}, KL {kl2:e}, топ-1 {top2}");
    let (m3, kl3, _, bits3) = compare(&lb, &lc);
    println!("с нуля (точка) ↔ с нуля (512): бит в бит {bits3}, max|Δ| {m3:e}, KL {kl3:e}");

    // Декод после восстановления: снова от точки, граф прошлого хода (буферы те же).
    s.set_resume_at(Some(point));
    let (oa, sta, ra2) = p.generate_cached_streaming(&mut s, &prompt2, &cfg(gen), &mut |_: u32| true)?;
    let mut f = p.new_session(ctx)?;
    f.set_resume_at(Some(point));
    let (ob, stb, _) = p.generate_cached_streaming(&mut f, &prompt2, &cfg(gen), &mut |_: u32| true)?;
    let same = oa.iter().zip(&ob).take_while(|(a, b)| a == b).count();
    println!(
        "декод хода 2: из кэша {ra2}, {} ток. ({:.1} ток/с) против с нуля {:.1} ток/с; совпало {same} из {}",
        sta.new_tokens,
        sta.new_tokens as f64 * 1000.0 / sta.decode_ms.max(1) as f64,
        stb.new_tokens as f64 * 1000.0 / stb.decode_ms.max(1) as f64,
        ob.len()
    );
    println!("ответ 2: {:?}", p.decode(&oa).unwrap_or_default());
    assert!(bits, "логиты после восстановления не бит в бит");
    assert_eq!(oa, ob, "декод после восстановления разошёлся");

    // Переезд в RAM и обратно: снимок едет вместе с кэшем, ход 3 снова от точки.
    let moved = s.park_to_host()?;
    let back = s.unpark_to(Device::Cuda(0))?;
    s.set_resume_at(Some(point));
    let (ld, rd) = p.prefill_cached(&mut s, &prompt2)?;
    let (m4, _, _, bits4) = compare(&host(&ld), &lb);
    println!("park {} МБ / unpark {} МБ, из кэша {rd}, бит в бит {bits4} (max|Δ| {m4:e}), снимок {:?}", moved >> 20, back >> 20, s.snapshot_info());
    assert!(bits4);
    println!("OK");
    Ok(())
}

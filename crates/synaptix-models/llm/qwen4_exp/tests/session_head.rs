//! Голова промпта в сессии префикс-KV (парковка контекста ролей): ходы с одной
//! головой и разными хвостами берут голову из кэша — каждый раз, а не только
//! первый, — и отвечают так же, как без кэша.
//!
//! Точка возврата ставится на конец головы (`set_resume_at`). Ход, взявший
//! голову из кэша, обязан оставить точку на ней: иначе она уезжает на конец
//! его промпта, и следующий ход с другим хвостом идёт в полный префилл.
//!
//! Модель — маленький эталон со случайными весами (`gen_qwen4_exp_ref.py`),
//! считается на CPU: `SYN_QWEN4EXP_REF=<каталог> cargo test -p synaptix-llm-qwen4-exp --test session_head`.

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_llm_common::generate::GenerationConfig;
use synaptix_llm_qwen4_exp::Qwen4ExpPipeline;

fn ref_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SYN_QWEN4EXP_REF").ok()?);
    p.join("model.safetensors").exists().then_some(p)
}

fn cfg() -> GenerationConfig {
    GenerationConfig { max_new_tokens: 8, ..GenerationConfig::default() }
}

/// Детерминированные «токены» в словаре 128 (0 — без особого смысла).
fn tokens(seed: u32, n: usize) -> Vec<u32> {
    let mut x = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            3 + x % 120
        })
        .collect()
}

fn plain(p: &Qwen4ExpPipeline, prompt: &[u32]) -> Vec<u32> {
    let mut s = p.new_session(256).expect("сессия");
    p.generate_cached_streaming(&mut s, prompt, cfg(), &mut |_: u32| true).expect("ход").0
}

#[test]
fn head_reused_every_turn() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
    let p = Qwen4ExpPipeline::load(&dir, Device::Cpu, DType::F32).expect("модель");
    let head = tokens(1, 70);
    let cat = |tail: &[u32]| [head.as_slice(), tail].concat();
    let (a, b, c) = (cat(&tokens(2, 13)), cat(&tokens(3, 9)), cat(&tokens(4, 21)));
    let (ref_a, ref_b, ref_c) = (plain(&p, &a), plain(&p, &b), plain(&p, &c));

    let mut s = p.new_session(256).expect("сессия");
    let mut turn = |prompt: &[u32]| {
        s.set_resume_at(Some(head.len()));
        let (out, _, reuse) = p.generate_cached_streaming(&mut s, prompt, cfg(), &mut |_: u32| true).expect("ход");
        (out, reuse)
    };
    // Первый ход считает всё и ставит точку на конец головы.
    assert_eq!(turn(&a), (ref_a.clone(), 0));
    // Дальше голова — из кэша на каждом ходу, ответы как без кэша.
    assert_eq!(turn(&b), (ref_b.clone(), head.len()));
    assert_eq!(turn(&c), (ref_c.clone(), head.len()));
    assert_eq!(turn(&a), (ref_a.clone(), head.len()));
    // Продолжение того же разговора (промпт + ответ + новое) идёт с живого
    // кэша — дальше головы, — и точка всё равно остаётся на голове.
    let cont = [a.as_slice(), ref_a.as_slice(), &tokens(5, 6)].concat();
    let ref_cont = plain(&p, &cont);
    let (out, reuse) = turn(&cont);
    assert_eq!(out, ref_cont);
    assert!(reuse > a.len(), "продолжение должно взять живой кэш: {reuse}");
    assert_eq!(turn(&b), (ref_b, head.len()));
}

/// Ход, не совпавший с кэшем ни на токен (полный префилл по старой сессии),
/// обязан ответить так же, как свежая сессия: сброс не оставляет состояния
/// прошлых ходов.
#[test]
fn full_prefill_on_used_session_matches_fresh() {
    synaptix_kernels_cpu::ensure_registered();
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
    let p = Qwen4ExpPipeline::load(&dir, Device::Cpu, DType::F32).expect("модель");
    let (a, b) = (tokens(11, 60), tokens(12, 50));
    let ref_b = plain(&p, &b);
    let mut s = p.new_session(256).expect("сессия");
    let (_, _, r) = p.generate_cached_streaming(&mut s, &a, cfg(), &mut |_: u32| true).expect("ход");
    assert_eq!(r, 0);
    let (out, _, r) = p.generate_cached_streaming(&mut s, &b, cfg(), &mut |_: u32| true).expect("ход");
    assert_eq!(r, 0);
    assert_eq!(out, ref_b, "полный префилл по использованной сессии разошёлся со свежей");
}

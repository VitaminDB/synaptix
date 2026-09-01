//! Префикс-KV Qwen4Exp на ЧАТОВОМ пути: политика `optimal` (KV MXFP8, как в
//! synthos), мультиходовой диалог с растущей историей — ход, продолживший с
//! кэша сессии, обязан выдать те же токены, что ход с полным префиллом.
//!
//! Существующий `qwen4exp_prefix_kv` гоняет `QuantPolicy::balance()` (KV F16)
//! и короткие промпты; чат же работает на `optimal_profile` → KV MXFP8 с
//! масштабом на 32 элемента и индексатором, сворачивающим ключи в блоки.
//! Откат такого кэша «по позиции» на небайтовой границе — отдельный путь,
//! который здесь и проверяется.
//!
//! Запуск: `SYN_QWEN4EXP_BUNDLE=… SYN_QWEN4EXP_EXPERT_CACHE_GB=6 cargo test
//! -p synaptix --release --test qwen4exp_prefix_kv_chat -- --nocapture
//! --test-threads=1`. Без переменной — no-op.

use std::path::Path;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, Device, GenerationOptions, LlmGeneration,
};

fn opts(max_seq_len: usize, max_new: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: max_new,
        max_seq_len,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 7,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    }
}

/// Ходы как в живом чате: системный промпт + задача, затем каждый ход
/// дописывает «ответ ассистента» и длинный «результат инструмента». Длины
/// подобраны под журнал упавшей сессии (1.5k → 4.7k → 5.3k токенов), а стыки
/// нарочно НЕ выровнены ни по 32 (масштаб MXFP8), ни по 64 (чанк GDN-скана).
fn chat_turns() -> Vec<String> {
    let system = "You are Syn, a local AI agent in the Synthos app. You run on the \
                  user's machine. Environment: linux, today 2026-09-01. Tools: bash, \
                  subagent. Rules: announce actions briefly, then call the tool. "
        .to_string();
    let task = "Задача: в проекте /home/master/Projects/2027/quitsmoke добавить режим \
                отказа от вейпа — второй счётчик на первой странице, дублирование всей \
                статистики и настройка в параметрах. Начни с осмотра проекта. ";
    let tool_result_a = "$ find /home/master/Projects/2027/quitsmoke -type f | head; \
                         exit: 0; src/main.rs src/model/stats.rs src/model/taper.rs \
                         src/screens/dashboard.rs src/screens/settings.rs Cargo.toml \
                         README.md docs/plan.md assets/icons.svg tests/smoke.rs "
        .repeat(40);
    let tool_result_b = "  138 src/model/achievements.rs\n  211 src/model/health.rs\n   \
                         66 src/model/log.rs\n   17 src/model/mod.rs\n   69 \
                         src/model/profile.rs\n  119 src/model/stats.rs\n  212 \
                         src/model/taper.rs\n"
        .repeat(12);
    let tool_result_c = "$ cat src/model/stats.rs; exit: 0; pub struct DayStats { pub \
                         smoked: u32, pub craved: u32, pub saved_money: f64 } impl \
                         DayStats { pub fn merge(&mut self, other: &DayStats) { \
                         self.smoked += other.smoked; } } "
        .repeat(18);

    let t0 = format!("{system}{task}");
    let t1 = format!("{t0}Assistant: смотрю структуру проекта. [bash] Result: {tool_result_a} ");
    let t2 = format!("{t1}Assistant: считаю строки модулей. [bash] Result: {tool_result_b} ");
    let t3 = format!("{t2}Assistant: читаю модель статистики. [bash] Result: {tool_result_c} ");
    vec![t0, t1, t2, t3]
}

#[test]
fn qwen4exp_prefix_kv_matches_full_prefill_on_chat_policy() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);

    // A/B по политике: `optimal` (KV MXFP8, как в чате) против `balance`
    // (KV F16) — расхождение только на optimal указывает на откат
    // квантованного KV, на обеих — на состояние, не зависящее от политики
    // (индексатор QSA, GDN).
    let policy = match std::env::var("SYN_TEST_POLICY").as_deref() {
        Ok("balance") => synaptix::facade::llm::QuantPolicy::balance(),
        _ => optimal_profile(Path::new(&path)).policy,
    };
    eprintln!("политика: preset={} kv={:?}", policy.preset_name, policy.kv_dtype);
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), policy, &Device::Cuda(0)).expect("load");

    // Токены каждого хода: хвост дописывается в токенах, чтобы префиксность
    // не ломалась о слияние на стыке.
    let turns = chat_turns();
    let mut ids: Vec<Vec<u32>> = Vec::new();
    for (i, t) in turns.iter().enumerate() {
        let full = tok.encode(t).expect("encode");
        match ids.last() {
            None => ids.push(full),
            Some(prev) => {
                // Хвост = новая часть текста, закодированная отдельно.
                let tail_text = &t[turns[i - 1].len()..];
                let mut v = prev.clone();
                v.extend_from_slice(&tok.encode(tail_text).expect("encode tail"));
                ids.push(v);
            }
        }
    }
    for (i, v) in ids.iter().enumerate() {
        eprintln!(
            "ход {i}: {} ток (стык {} — mod 32 = {}, mod 64 = {})",
            v.len(),
            if i > 0 { ids[i - 1].len() } else { 0 },
            if i > 0 { ids[i - 1].len() % 32 } else { 0 },
            if i > 0 { ids[i - 1].len() % 64 } else { 0 },
        );
    }

    let ctx = 16384usize;
    let max_new = 48usize;

    fn first_diff(a: &[u32], b: &[u32]) -> Option<usize> {
        if a == b {
            return None;
        }
        Some(
            a.iter()
                .zip(b)
                .position(|(x, y)| x != y)
                .unwrap_or_else(|| a.len().min(b.len())),
        )
    }

    // A. Эталон: каждый ход с нуля, полный префилл — ДВАЖДЫ. Стек не
    // бит-детерминирован (порядок редукций в ядрах плавает), поэтому паре
    // fresh-прогонов позволено расходиться; их точка расхождения — базис
    // «числового джиттера», с которым сравнивается кэшированный ход.
    let mut fresh: Vec<Vec<u32>> = Vec::new();
    let mut fresh_jitter: Vec<Option<usize>> = Vec::new();
    for v in &ids {
        let mut runs: Vec<Vec<u32>> = Vec::new();
        for _ in 0..2 {
            let mut got = Vec::new();
            let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
            r.generate_streaming(v, &tok, |id, _| {
                got.push(id);
                true
            })
            .expect("fresh turn");
            runs.push(got);
        }
        fresh_jitter.push(first_diff(&runs[0], &runs[1]));
        fresh.push(runs.swap_remove(0));
    }
    for (i, j) in fresh_jitter.iter().enumerate() {
        eprintln!("ход {i}: fresh-vs-fresh джиттер = {j:?}");
    }

    // B. Чатовый путь: ДВЕ независимые сессии проходят тот же диалог.
    // C1-vs-C2 — собственная стабильность пути с рестором; сравнение её с
    // fresh-джиттером и отвечает на вопрос «портит ли restore состояние»:
    // исправный restore шумит не сильнее самой арифметики ядер.
    let mut cached: Vec<Vec<Vec<u32>>> = vec![Vec::new(), Vec::new()];
    for c in 0..2 {
        let mut session = model
            .new_kv_session(ctx, max_new)
            .expect("session")
            .expect("Qwen4Exp умеет префикс-KV");
        for (i, v) in ids.iter().enumerate() {
            let mut got = Vec::new();
            let mut r = LlmGeneration::new(&model, opts(ctx, max_new));
            let reused = r
                .generate_streaming_cached(&mut session, v, &tok, |id, _| {
                    got.push(id);
                    true
                })
                .expect("cached turn");
            let expect_reuse = if i == 0 { 0 } else { ids[i - 1].len() };
            assert_eq!(reused, expect_reuse, "сессия {c}, ход {i}: переиспользовано не столько");
            cached[c].push(got);
        }
    }
    let mut catastrophic = Vec::new();
    for i in 0..ids.len() {
        let vs_fresh = first_diff(&fresh[i], &cached[0][i]);
        let self_jitter = first_diff(&cached[0][i], &cached[1][i]);
        eprintln!(
            "ход {i}: fresh-vs-cached {vs_fresh:?}, cached-vs-cached {self_jitter:?}, \
             fresh-vs-fresh {:?}",
            fresh_jitter[i]
        );
        eprintln!(
            "  fresh:  {:?}\n  cached: {:?}",
            tok.decode(&fresh[i]).unwrap_or_default(),
            tok.decode(&cached[0][i]).unwrap_or_default()
        );
        // Катастрофа образца бага хвоста индексатора: кэшированный ход
        // рассыпается сразу (первые токены), пока fresh-пара ещё стабильна.
        // Тонкие расхождения дальше по хвосту — джиттер ядер, их судит
        // юнит-тест `indexer_tail_roundtrip`, а не эта проба.
        if let Some(pos) = vs_fresh {
            let stable_to = fresh_jitter[i].unwrap_or(usize::MAX);
            if pos + 8 < stable_to {
                catastrophic.push((i, pos, stable_to));
            }
        }
    }
    assert!(
        catastrophic.is_empty(),
        "ходы {catastrophic:?}: кэшированный путь рассыпался там, где fresh стабилен — \
         порча состояния сессии (см. IndexerCache::tail_snapshot)"
    );
}

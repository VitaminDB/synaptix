//! Диагностика: калечит ли repeat_penalty числовые константы при повторе.
//!
//! Сценарий из живого чата 02.09.2026: агент правит sed'ом собственный тест и
//! обязан написать `1_000_000 + 3 * 86_400_000` дважды (паттерн + замена) в
//! окне repeat_last_n. Гипотеза: штраф за повтор сбивает digit-токены второй
//! копии, число выходит усечённым (`86_4`, `864`, `1_0`…), правка не
//! срабатывает — агент зацикливается.
//!
//! Запуск: `SYN_QWEN4EXP_BUNDLE=… SYN_QWEN4EXP_EXPERT_CACHE_GB=6 cargo test
//! -p synaptix --release --test qwen4exp_digit_penalty_probe -- --nocapture
//! --test-threads=1`. Без переменной — no-op.

use std::path::Path;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, Device, GenerationOptions, LlmGeneration,
};

fn opts(repeat_penalty: f32) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: 96,
        max_seq_len: 4096,
        temperature: 0.0,
        top_k: 20,
        top_p: 0.9,
        min_p: 0.0,
        seed: 7,
        repeat_penalty,
        repeat_last_n: 64,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    }
}

#[test]
fn digits_survive_repeat_penalty() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);

    let profile = optimal_profile(Path::new(&path));
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), profile.policy, &Device::Cuda(0)).expect("load");

    let needle = "1_000_000 + 3 * 86_400_000";
    let prompt = format!(
        "Print this exact line two times, nothing else:\n\
         assert_eq!(p.health_baseline(), {needle});\n\nOutput:\n"
    );
    let ids = tok.encode(&prompt).expect("encode");

    let mut outputs = Vec::new();
    for rp in [1.05f32, 1.0] {
        let mut got = Vec::new();
        let mut r = LlmGeneration::new(&model, opts(rp));
        r.generate_streaming(&ids, &tok, |id, _| {
            got.push(id);
            true
        })
        .expect("generate");
        let text = tok.decode(&got).unwrap_or_default();
        let hits = text.matches(needle).count();
        eprintln!("rp={rp}: константа воспроизведена {hits} раз(а) из 2\n---\n{text}\n---");
        outputs.push((rp, hits));
    }
    // Жёстко проверяем только чистый режим: без штрафа greedy обязан
    // воспроизвести константу оба раза. Поведение rp=1.05 — диагностика.
    let clean = outputs.iter().find(|(rp, _)| *rp == 1.0).unwrap();
    assert_eq!(clean.1, 2, "rp=1.0: greedy обязан воспроизвести константу дважды");
}

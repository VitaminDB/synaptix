//! Бенч Qwen4Exp (qwen3.8-flash-next) по сценарию чата: длинная история +
//! ходы с префикс-KV, у каждого хода — хвост промпта и декод.
//!
//! Запуск:
//!   SYN_QWEN4EXP_BUNDLE=…/qwen3.8-flash-next.syn SYN_QWEN4EXP_EXPERT_CACHE_GB=10 \
//!   SYN_BENCH_CTX=80000 SYN_BENCH_TAIL=3000 SYN_BENCH_NEW=64 SYN_BENCH_TURNS=3 \
//!   [SYN_QWEN4EXP_PROFILE=sync SYN_MOE_PROFILE=1] \
//!   cargo test -p synaptix --release --test qwen4exp_bench -- --nocapture --test-threads=1
//!
//! Печатает на каждый ход: длина промпта, переиспользовано из кэша, префилл
//! (время до первого токена), декод ток/с; в конце — профили этапов.

use std::path::Path;
use std::time::Instant;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, Device, GenerationOptions, LlmGeneration,
};

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

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

/// Текст «результата инструмента» примерно на `want` токенов: разнообразные
/// абзацы, чтобы роутер MoE не вырождался на повторе.
fn filler(tok: &synaptix::facade::llm::LlmTokenizer, want: usize, seed: usize) -> Vec<u32> {
    let paras = [
        "Файл src/model/stats.rs: pub struct DayStats { smoked: u32, craved: u32, saved: f64 } \
         impl DayStats { pub fn merge(&mut self, o: &DayStats) { self.smoked += o.smoked; } }\n",
        "Журнал сборки: cargo build --release, 214 крейтов, предупреждений 37, ошибок 0, \
         время 2 мин 41 с; бинарь target/release/app, 22 МБ после strip.\n",
        "Заметка: в понедельник сверить остатки на складе с учётной системой, обновить \
         сертификаты на шлюзе, проверить резервные копии за неделю и журналы доступа.\n",
        "Таблица: | id | имя | статус | срок |\n| 17 | миграция базы | в работе | 12.09 |\n\
         | 18 | обновление UI | готово | 09.09 |\n| 19 | тесты API | очередь | 15.09 |\n",
        "def parse(line: str) -> dict:\n    key, _, value = line.partition('=')\n    \
         return {key.strip(): value.strip()}\n",
        "Ошибка: connection refused (os error 111) при обращении к 127.0.0.1:8080; \
         повтор через 5 с, попытка 3 из 5; служба перезапущена systemd, статус active.\n",
    ];
    let mut text = String::new();
    let mut i = seed;
    loop {
        text.push_str(&format!("[{i}] "));
        text.push_str(paras[i % paras.len()]);
        i += 1;
        if i % 32 == 0 {
            let n = tok.encode(&text).expect("encode").len();
            if n >= want {
                break;
            }
        }
    }
    let mut ids = tok.encode(&text).expect("encode");
    ids.truncate(want);
    ids
}

#[test]
fn qwen4exp_chat_bench() {
    let Ok(path) = std::env::var("SYN_QWEN4EXP_BUNDLE") else {
        eprintln!("SYN_QWEN4EXP_BUNDLE не задан — пропускаем");
        return;
    };
    let ctx = env_usize("SYN_BENCH_CTX", 8000);
    let tail = env_usize("SYN_BENCH_TAIL", 3000);
    let max_new = env_usize("SYN_BENCH_NEW", 64);
    let turns = env_usize("SYN_BENCH_TURNS", 3);

    synaptix::facade::llm::cuda_release_kernel_caches();
    synaptix::facade::llm::cuda_trim_pool(0);
    let policy = optimal_profile(Path::new(&path)).policy;
    eprintln!("политика: preset={} kv={:?}", policy.preset_name, policy.kv_dtype);
    let t = Instant::now();
    let (model, tok) =
        load_llm_with_policy(Path::new(&path), policy, &Device::Cuda(0)).expect("load");
    eprintln!("загрузка: {:.1} с", t.elapsed().as_secs_f64());

    let head = tok
        .encode("<|im_start|>system\nYou are Syn, a local AI agent.<|im_end|>\n<|im_start|>user\nПрочитай журнал и ответь, сколько записей со статусом «готово».\n")
        .expect("encode");
    let mut ids = head.clone();
    ids.extend(filler(&tok, ctx.saturating_sub(head.len()), 0));
    let ask = tok
        .encode("<|im_end|>\n<|im_start|>assistant\n")
        .expect("encode");

    // SYN_BENCH_SESSION — ёмкость сессии как в чате (synthos берёт её с
    // двойным запасом, до cap модели 262143), чтобы воспроизвести пик VRAM.
    let max_seq = env_usize("SYN_BENCH_SESSION", ctx + tail * turns + max_new * (turns + 1) + 512);
    let mut session = model
        .new_kv_session(max_seq, max_new)
        .expect("session")
        .expect("Qwen4Exp умеет префикс-KV");

    // SYN_BENCH_WAIT_MIRROR=1 — дождаться зеркала экспертов в RAM перед
    // первым ходом (тёплый замер подкачки).
    if std::env::var("SYN_BENCH_WAIT_MIRROR").is_ok() {
        let t = Instant::now();
        while let Some((ready, total)) = synaptix_core::device::cuda::offload_pin_cache_progress() {
            if ready >= total || t.elapsed().as_secs() > 600 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(500));
        }
        eprintln!(
            "зеркало: {:?} за {:.1} с",
            synaptix_core::device::cuda::offload_pin_cache_progress().map(|(r, t)| (r >> 20, t >> 20)),
            t.elapsed().as_secs_f64()
        );
    }
    let mut last_answer: Vec<u32> = Vec::new();
    for turn in 0..=turns {
        if turn > 0 {
            // Как в чате: новый промпт = прежний промпт целиком + реплика
            // ассистента + результат инструмента (прежний промпт — префикс).
            ids.extend_from_slice(&ask);
            ids.extend_from_slice(&last_answer);
            ids.extend(tok.encode("<|im_end|>\n<|im_start|>user\n").expect("encode"));
            ids.extend(filler(&tok, tail, turn * 100_000));
        }
        let mut prompt = ids.clone();
        prompt.extend_from_slice(&ask);
        let mut r = LlmGeneration::new(&model, opts(max_seq, max_new));
        let t0 = Instant::now();
        let mut first: Option<Instant> = None;
        let mut n = 0usize;
        let mut text = String::new();
        let mut answer_ids: Vec<u32> = Vec::new();
        let reused = r
            .generate_streaming_cached(&mut session, &prompt, &tok, |id, s| {
                if first.is_none() {
                    first = Some(Instant::now());
                }
                n += 1;
                text.push_str(s);
                answer_ids.push(id);
                true
            })
            .expect("turn");
        last_answer = answer_ids;
        let total = t0.elapsed().as_secs_f64();
        let first_at = first.map(|f| f.duration_since(t0).as_secs_f64()).unwrap_or(total);
        let decode = (total - first_at).max(1e-6);
        eprintln!(
            "ход {turn}: промпт {} ток, из кэша {reused}, префилл {:.0} мс ({} ток → {:.0} ток/с), \
             декод {n} ток за {:.2} с = {:.1} ток/с; всего {:.1} с",
            prompt.len(),
            first_at * 1000.0,
            prompt.len() - reused,
            (prompt.len() - reused) as f64 / first_at,
            decode,
            (n.saturating_sub(1)) as f64 / decode,
            total
        );
        eprintln!("  ответ: {:?}", text.chars().take(120).collect::<String>());
        if let Some((r, t)) = synaptix_core::device::cuda::offload_pin_cache_progress() {
            eprintln!("  зеркало экспертов: {} / {} МБ", r >> 20, t >> 20);
        }
        if std::env::var("SYN_QWEN4EXP_PROFILE").is_ok() {
            eprintln!("{}", synaptix_llm_qwen4_exp::norm::profile_report());
        }
        if std::env::var("SYN_MOE_PROFILE").is_ok() {
            eprintln!("{}", synaptix_llm_common::profile::report());
        }
    }
    if std::env::var("SYN_QWEN4EXP_PROFILE").is_ok() {
        eprintln!("{}", synaptix_llm_qwen4_exp::norm::profile_report());
    }
    if std::env::var("SYN_MOE_PROFILE").is_ok() {
        eprintln!("{}", synaptix_llm_common::profile::report());
    }
}

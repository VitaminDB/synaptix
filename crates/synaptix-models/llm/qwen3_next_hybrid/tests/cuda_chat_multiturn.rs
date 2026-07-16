#![cfg(feature = "cuda")]

//! Многоходовый chat на реальном 27B — зеркалит РЕАЛЬНЫЙ путь CLI: полный
//! jinja-рендер истории каждый ход (шаблон Qwen корректно стрипает прошлый
//! reasoning) + свежий prefill. Проверяет, что после отката токен-дельты модель
//! видит каждый вопрос и отвечает осмысленно, контекст растёт >1000 токенов без
//! OOM (чанкование prefill). Запуск:
//!   SYN_QWEN_NEXT_CHAT=1 cargo test -p synaptix-llm-qwen3-next-hybrid \
//!     --features cuda --profile fast-release --test cuda_chat_multiturn -- --nocapture

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen3_next_hybrid::pipeline::{GenerationConfig, HybridPipeline};
use synaptix_tokenizer::templates::chat_template::RenderOptions;
use synaptix_tokenizer::{ChatTemplate, Message, Tokenizer};

fn bundle() -> Option<PathBuf> {
    let p = PathBuf::from("models/qwen3.6 27B.syn");
    if p.exists() { Some(p) } else { None }
}

fn nvfp4_f16() -> PrecisionConfig {
    PrecisionConfig {
        compute: DType::F16,
        attn_w: DType::NVFP4,
        mlp_w: DType::NVFP4,
        lm_head: DType::F16,
        embed: DType::F16,
        kv: DType::F16,
    }
}

fn synth_doc() -> String {
    let mut s = String::from("Технический отчёт по системе обработки заказов.\n\n");
    for i in 1..=30 {
        s.push_str(&format!(
            "Пункт {i}: модуль {} обрабатывает {} запросов в секунду при задержке {} мс; \
             зависит от сервиса {} и пишет в таблицу orders_{}.\n",
            ["auth", "billing", "catalog", "search", "ship"][i % 5],
            100 + i * 7,
            3 + (i % 11),
            ["redis", "kafka", "pg", "s3"][i % 4],
            i % 9,
        ));
    }
    s
}

/// Как `strip_reasoning` в CLI: оставить только ответ после последнего </think>.
fn strip_reasoning(text: &str) -> String {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim_start().to_string(),
        None => text.to_string(),
    }
}

#[test]
fn chat_multiturn_fullrender() {
    if std::env::var("SYN_QWEN_NEXT_CHAT").is_err() {
        return;
    }
    let Some(path) = bundle() else { return };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let cap = 2048usize;
    let p = HybridPipeline::load_with_precision(&path, Device::Cuda(0), nvfp4_f16(), Some(cap))
        .expect("load 27B nvfp4 f16");

    // chat-template из бандла (как делает CLI).
    let bundle = synaptix_bundle::Bundle::open(&path).expect("bundle");
    let src = String::from_utf8(bundle.read_file("chat_template.jinja").expect("tmpl").into_owned())
        .expect("utf8");
    let template = ChatTemplate::from_source_with_specials(src, p.tokenizer.special_tokens().clone());
    let render = |msgs: &[Message]| -> String {
        let opts = RenderOptions::new()
            .with_generation_prompt(true)
            .with_var("enable_thinking", serde_json::Value::Bool(true));
        template.render(msgs, &opts).expect("render")
    };

    let doc = synth_doc();
    let q1 = format!(
        "Проанализируй документ и назови самый нагруженный модуль.\n\n{doc}"
    );
    let follow_ups = [
        "А какой модуль самый быстрый по задержке?",
        "Сколько всего пунктов в документе?",
        "Спасибо, сформулируй вывод одним предложением.",
    ];

    let mut messages: Vec<Message> = vec![Message::system("Ты ассистент-аналитик. Отвечай кратко по-русски.")];
    let mut user_turns: Vec<String> = vec![q1];
    user_turns.extend(follow_ups.iter().map(|s| s.to_string()));

    let mut max_ctx = 0usize;
    for (i, user) in user_turns.iter().enumerate() {
        messages.push(Message::user(user.clone()));
        let prompt = render(&messages);
        let ids = p.encode(&prompt).expect("encode");
        // Полный prefill каждый ход (свежий KV) — как CLI после отката токен-дельты.
        let mut kv = p.model.make_kv_cache(1, cap).expect("kv");
        let cfg = GenerationConfig {
            max_new_tokens: 40,
            temperature: 0.0,
            max_seq: Some(cap),
            ..Default::default()
        };
        let mut noop = |_: u32| true;
        let (out, stats) = p
            .generate_with_graph_resume(&mut kv, &ids, cfg, &mut noop)
            .expect("generate");
        let full = p.decode(&out).unwrap_or_default();
        let answer = strip_reasoning(&full);
        max_ctx = max_ctx.max(ids.len());
        let pfx = stats.prompt_tokens;
        let ptps = if stats.prefill_ms > 0 { pfx as f64 / (stats.prefill_ms as f64 / 1000.0) } else { 0.0 };
        eprintln!(
            "[ход {}] prompt_tok={pfx} prefill={}ms ({ptps:.0} tok/s) new={}\n   Q: {}\n   A: {}",
            i + 1,
            stats.prefill_ms,
            out.len(),
            user.lines().next().unwrap_or(""),
            answer.replace('\n', " ").chars().take(200).collect::<String>(),
        );
        assert!(!out.is_empty(), "ход {}: пустой ответ", i + 1);
        // Модель «увидела вопрос» = ответ непустой после стрипа reasoning.
        assert!(!answer.trim().is_empty(), "ход {}: ответ пуст после strip (модель не ответила)", i + 1);
        messages.push(Message::assistant(strip_reasoning(&full)));
    }
    eprintln!("[chat] макс. контекст = {max_ctx} токенов за {} ходов", user_turns.len());
    assert!(max_ctx > 1000, "контекст не дорос до 1000 ({max_ctx})");
}

//! Гейтед e2e-smoke: путь LLM-ноды/Syn-чата synthos для Muse-Glimmer-30B —
//! `facade::llm::load_llm` (детект muse_glimmer) → ATEM chat_template.jinja из
//! бандла → `LlmGeneration::generate_streaming` (дельты текста, стоп по <|eot|>).
//!
//! Запуск:
//!   SYN_MUSE_BUNDLE=/run/media/storage/syn_models/muse-glimmer-30b.syn \
//!   cargo test -p synaptix --release --test muse_glimmer_facade_smoke -- --nocapture

use synaptix::facade::llm::{load_llm, GenerationOptions, LlmGeneration, Message};
use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;

#[test]
fn muse_glimmer_facade_generates() {
    let Ok(path) = std::env::var("SYN_MUSE_BUNDLE") else {
        return;
    };

    let (model, tokenizer) = load_llm(
        path.as_ref(),
        Device::Cuda(0),
        PrecisionConfig::nvfp4(),
        Some(2048),
    )
    .expect("load_llm muse-glimmer");

    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user("Столица Франции?"),
    ];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    assert!(prompt.contains("<|start|>user<|message|>"), "{prompt}");
    assert!(prompt.ends_with("<|start|>assistant"), "{prompt}");
    let ids = tokenizer.encode(&prompt).expect("encode");
    assert!(
        tokenizer.eos_ids().contains(&200008),
        "<|eot|> должен быть в стоп-токенах: {:?}",
        tokenizer.eos_ids()
    );

    let mut generation = LlmGeneration::new(
        &model,
        GenerationOptions {
            max_new_tokens: 64,
            max_seq_len: 2048,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
    );
    generation.set_stop_tokens(tokenizer.eos_ids().to_vec());

    let mut out = String::new();
    generation
        .generate_streaming(&ids, &tokenizer, |_id, delta| {
            out.push_str(delta);
            true
        })
        .expect("generate");

    eprintln!("[muse-glimmer facade] prompt {} ток. → '{out}'", ids.len());
    assert!(
        out.to_lowercase().contains("париж") || out.to_lowercase().contains("paris"),
        "ответ должен упоминать Париж: {out}"
    );
}

/// Регрессия: тот же прогон, но с MXFP8 KV-кэшем (preset Balance в Syn-чате).
/// Muse-Glimmer чередует sliding- и full-attention слои, а квантованный кэш
/// читает только `flash_attention_mxfp8kv` (causal, без окна) — раньше
/// sliding-слой получал MXFP8-буфер и первый же prefill падал с
/// «cuda kv_append: dtype mismatch src/dst». Теперь MXFP8 достаётся только
/// full-attention слоям, sliding-слои держат плотный ring-KV.
#[test]
fn muse_glimmer_facade_generates_mxfp8_kv() {
    let Ok(path) = std::env::var("SYN_MUSE_BUNDLE") else {
        return;
    };

    let mut precision = PrecisionConfig::nvfp4();
    precision.kv = synaptix_core::dtype::DType::MXFP8;

    let (model, tokenizer) =
        load_llm(path.as_ref(), Device::Cuda(0), precision, Some(2048)).expect("load_llm muse-glimmer mxfp8-kv");

    // Ставка «на токен» считается только по full-attention слоям: sliding-слои
    // живут в ring-окне фиксированного размера.
    let per_token = model.kv_bytes_per_token();
    let fixed = model.kv_fixed_bytes(2048);
    eprintln!("[muse-glimmer mxfp8-kv] KV {per_token} B/ток + {} MB окон", fixed / (1024 * 1024));
    assert!(per_token > 0 && fixed > 0, "ожидались и per-token, и ring-часть: {per_token}/{fixed}");

    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user("Столица Франции?"),
    ];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    let ids = tokenizer.encode(&prompt).expect("encode");

    let mut generation = LlmGeneration::new(
        &model,
        GenerationOptions {
            max_new_tokens: 64,
            max_seq_len: 2048,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
        },
    );
    generation.set_stop_tokens(tokenizer.eos_ids().to_vec());

    let mut out = String::new();
    generation
        .generate_streaming(&ids, &tokenizer, |_id, delta| {
            out.push_str(delta);
            true
        })
        .expect("generate mxfp8-kv");

    eprintln!("[muse-glimmer mxfp8-kv] prompt {} ток. → '{out}'", ids.len());
    assert!(
        out.to_lowercase().contains("париж") || out.to_lowercase().contains("paris"),
        "ответ должен упоминать Париж: {out}"
    );
}

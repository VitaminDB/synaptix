//! Гейтед e2e-smoke: путь LLM-ноды/Syn-чата synthos для Qwen3.8-27B —
//! `facade::llm::load_llm` (детект qwen3_5 → Hybrid) → chat_template.jinja из
//! бандла → `LlmGeneration::generate_streaming` (дельты текста, стоп по EOS).
//!
//! Запуск:
//!   SYN_QWEN38_BUNDLE=/run/media/storage/syn_models/qwen3.8-27b.syn \
//!   cargo test -p synaptix --release --test qwen38_facade_smoke -- --nocapture

use synaptix::facade::llm::{load_llm, GenerationOptions, LlmGeneration, Message};
use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;

#[test]
fn qwen38_facade_generates() {
    let Ok(path) = std::env::var("SYN_QWEN38_BUNDLE") else {
        return;
    };

    let (model, tokenizer) = load_llm(
        path.as_ref(),
        Device::Cuda(0),
        PrecisionConfig::nvfp4(),
        Some(2048),
    )
    .expect("load_llm qwen3.8");

    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user("Столица Франции?"),
    ];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    assert!(prompt.contains("<|im_start|>user"), "{prompt}");
    let ids = tokenizer.encode(&prompt).expect("encode");

    let mut generation = LlmGeneration::new(
        &model,
        GenerationOptions {
            max_new_tokens: 48,
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
    generation.generate_streaming(&ids, &tokenizer, |_id, delta| {
        out.push_str(delta);
        true
    })
    .expect("generate");

    eprintln!("[qwen3.8 facade] prompt {} ток. → '{out}'", ids.len());
    assert!(
        out.to_lowercase().contains("париж"),
        "ответ должен упоминать Париж: {out}"
    );
}

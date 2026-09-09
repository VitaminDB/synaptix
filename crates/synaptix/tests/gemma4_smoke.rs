//! Гейтед e2e-smoke Gemma-4 26B A4B: `.syn`-бандл (NVFP4-эксперты) → пайплайн
//! → greedy-генерация на сыром промпте в разметке модели.
//!
//! Промпт собирается вручную, а не шаблоном: тест проверяет матчасть модели
//! (MoE-ветка, широкие global-слои с V=K, proportional RoPE), а не jinja.
//!
//! Запуск:
//!   SYN_GEMMA4_BUNDLE=/home/master/Storage/syn_models/gemma-4-26b-a4b-it.syn \
//!   cargo test -p synaptix --release --test gemma4_smoke -- --nocapture --test-threads=1

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_common::GenerationConfig;
use synaptix_llm_gemma4::Gemma4Pipeline;

fn bundle() -> Option<String> {
    std::env::var("SYN_GEMMA4_BUNDLE").ok()
}

/// Разметка хода Gemma-4 без «мыслей» (`enable_thinking=false`): пустой
/// канал мысли дописывается сразу за началом реплики модели.
fn prompt(system: &str, user: &str) -> String {
    format!(
        "<bos><|turn>system\n{system}<turn|>\n<|turn>user\n{user}<turn|>\n\
         <|turn>model\n<|channel>thought\n<channel|>"
    )
}

#[test]
fn gemma4_generates() {
    let Some(path) = bundle() else { return };
    synaptix::init().expect("init");
    let precision = PrecisionConfig {
        compute: DType::BF16,
        attn_w: DType::BF16,
        mlp_w: DType::NVFP4,
        lm_head: DType::BF16,
        embed: DType::BF16,
        kv: DType::BF16,
    };
    let t0 = std::time::Instant::now();
    let pipe = Gemma4Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(2048))
        .expect("load gemma4");
    eprintln!("[gemma4] загрузка {:?}", t0.elapsed());
    eprintln!(
        "[gemma4] слоёв {}, экспертов {}×{}, KV {} Б/ток",
        pipe.config.num_hidden_layers,
        pipe.config.num_experts,
        pipe.config.top_k_experts,
        pipe.model.kv_bytes_per_token()
    );

    let text = prompt("Отвечай кратко, одним предложением.", "Столица Франции?");
    // `<bos>` и прочая разметка уже в тексте — спецтокены добавлять не надо.
    let ids = {
        use synaptix_tokenizer::Tokenizer;
        pipe.tokenizer.encode(&text, false).expect("encode").ids
    };
    eprintln!("[gemma4] промпт {} ток.", ids.len());

    let cfg = GenerationConfig {
        max_new_tokens: 48,
        temperature: 0.0,
        max_seq: Some(2048),
        ..Default::default()
    };
    let t1 = std::time::Instant::now();
    let (new_ids, stats) = pipe.generate(&ids, cfg).expect("generate");
    let out = pipe.decode(&new_ids).expect("decode");
    eprintln!(
        "[gemma4] {} ток. за {:?} (prefill {} мс, decode {} мс) → {out:?}",
        new_ids.len(),
        t1.elapsed(),
        stats.prefill_ms,
        stats.decode_ms
    );
    assert!(!new_ids.is_empty(), "модель ничего не выдала");
    assert!(
        out.to_lowercase().contains("париж") || out.to_lowercase().contains("paris"),
        "ответ должен упоминать Париж: {out:?}"
    );
}

/// Длинный промпт: sliding-окно (1024) уже не покрывает начало текста,
/// поэтому иголку можно достать ТОЛЬКО через global-слои — те самые, у
/// которых широкая голова, общий K/V и proportional RoPE. Ошибка в них даёт
/// связный, но неверный ответ, и короткий промпт её не ловит.
#[test]
fn gemma4_finds_needle_past_sliding_window() {
    let Some(path) = bundle() else { return };
    synaptix::init().expect("init");
    let precision = PrecisionConfig {
        compute: DType::BF16,
        attn_w: DType::BF16,
        mlp_w: DType::NVFP4,
        lm_head: DType::BF16,
        embed: DType::BF16,
        kv: DType::BF16,
    };
    let pipe = Gemma4Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(4096))
        .expect("load gemma4");

    let mut hay = String::new();
    hay.push_str("Секретный код доступа к серверу — ЛИМОН-7429. Запомни его.\n\n");
    for i in 0..90 {
        hay.push_str(&format!(
            "Пункт {i}. Регламент обслуживания предписывает проверять журналы, \
             обновлять сертификаты и вести учёт заявок в общем реестре.\n"
        ));
    }
    let text = prompt(
        "Отвечай кратко, только по тексту.",
        &format!("{hay}\nКакой секретный код доступа к серверу упомянут в начале текста?"),
    );
    let ids = {
        use synaptix_tokenizer::Tokenizer;
        pipe.tokenizer.encode(&text, false).expect("encode").ids
    };
    assert!(ids.len() > 1200, "промпт должен быть длиннее окна: {}", ids.len());

    let cfg = GenerationConfig {
        max_new_tokens: 40,
        temperature: 0.0,
        max_seq: Some(4096),
        // Веса занимают 18 из 24 ГБ — префилл длинного промпта одним куском
        // не оставляет места активациям MoE.
        prefill_batch: 256,
        ..Default::default()
    };
    let (new_ids, stats) = pipe.generate(&ids, cfg).expect("generate");
    let out = pipe.decode(&new_ids).expect("decode");
    eprintln!(
        "[gemma4 needle] промпт {} ток., prefill {} мс → {out:?}",
        ids.len(),
        stats.prefill_ms
    );
    assert!(out.contains("7429"), "иголка не найдена: {out:?}");
}

/// Путь synthos целиком: `load_llm` (детект `gemma4`) → chat_template.jinja из
/// бандла → `LlmGeneration::generate_streaming`.
#[test]
fn gemma4_facade_with_chat_template() {
    let Some(path) = bundle() else { return };
    use synaptix::facade::llm::{load_llm, GenerationOptions, LlmGeneration, Message};

    let (model, tokenizer) = load_llm(
        path.as_ref(),
        Device::Cuda(0),
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::BF16,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        Some(2048),
    )
    .expect("load_llm gemma4");

    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user("Столица Японии?"),
    ];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    eprintln!("[gemma4 tpl] {prompt:?}");
    assert!(prompt.contains("<|turn>user"), "разметка хода: {prompt}");
    assert!(prompt.ends_with("<|turn>model\n<|channel>thought\n<channel|>"), "хвост: {prompt:?}");

    let ids = tokenizer.encode(&prompt).expect("encode");
    let mut generation = LlmGeneration::new(
        &model,
        GenerationOptions {
            max_new_tokens: 32,
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
    eprintln!("[gemma4 tpl] промпт {} ток. → {out:?}", ids.len());
    assert!(out.to_lowercase().contains("токио"), "ответ: {out:?}");
}

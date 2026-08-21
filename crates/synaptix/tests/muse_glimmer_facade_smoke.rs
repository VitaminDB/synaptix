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

/// Профиль Balance из Syn-чата: NVFP4-веса, MXFP8 lm_head/embed/KV, compute F16.
/// PrecisionConfig::nvfp4() держит embed в F16 — на vocab 202k это лишние
/// ~2.7 ГБ, и на 24 ГБ длинный prefill в них упирается.
fn balance_precision() -> PrecisionConfig {
    use synaptix_core::dtype::DType;
    PrecisionConfig {
        compute: DType::F16,
        attn_w: DType::NVFP4,
        mlp_w: DType::NVFP4,
        lm_head: DType::MXFP8,
        embed: DType::MXFP8,
        kv: DType::MXFP8,
    }
}

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

    let (model, tokenizer) = load_llm(path.as_ref(), Device::Cuda(0), balance_precision(), Some(2048))
        .expect("load_llm muse-glimmer mxfp8-kv");

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

/// Регрессия: промпт длиннее ring-окна драфтера. DFlash получает tap-hidden'ы
/// ВСЕГО промпта одним блоком, а его кэш рассчитан на окно
/// (sliding_window + RING_SLACK + block_size = 4112) — на промпте ~5k ход падал
/// с «cuda kv_append: seq_pos 0 + t_new 4980 > max_seq 4112». Контекст драфтера
/// режется по хвосту в одно окно (дальше band-маска всё равно не смотрит).
#[test]
fn muse_glimmer_long_prompt_dflash() {
    let Ok(path) = std::env::var("SYN_MUSE_BUNDLE") else {
        return;
    };

    // 24 ГБ едва хватает на веса 30B: prefill режем на 128, кэш — под ровно тот
    // контекст, который нужен тесту.
    synaptix::facade::llm::set_prefill_chunk_size(128);
    let (model, tokenizer) = load_llm(path.as_ref(), Device::Cuda(0), balance_precision(), Some(6144))
        .expect("load_llm muse-glimmer");
    // Как synthos после загрузки: вернуть драйверу слабину staging-пула, иначе
    // на 24 ГБ prefill упирается в OOM ещё до KV.
    let freed = synaptix::facade::llm::cuda_trim_pool(0);
    eprintln!("[muse-glimmer long] trim после загрузки: +{freed} MB");

    // ~5k токенов «протокола»: заведомо длиннее и ring-окна target'а (2048+2048),
    // и кэша драфтера (4112).
    let mut doc = String::new();
    for i in 1..=125 {
        doc.push_str(&format!(
            "Пункт {i}. Сервис mp3party отдаёт страницу со списком треков; \
             для каждого трека нужны название, исполнитель и прямая ссылка на mp3.\n"
        ));
    }
    let msgs = [
        Message::system("Отвечай кратко, одним предложением."),
        Message::user(format!("{doc}\nСколько пунктов в протоколе выше? Ответь числом.")),
    ];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    let ids = tokenizer.encode(&prompt).expect("encode");
    assert!(ids.len() > 4112, "промпт должен превышать кэш драфтера: {}", ids.len());

    let mut generation = LlmGeneration::new(
        &model,
        GenerationOptions {
            max_new_tokens: 48,
            max_seq_len: 6144,
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
        .expect("generate long prompt");

    eprintln!("[muse-glimmer long] prompt {} ток. → '{out}'", ids.len());
    assert!(!out.trim().is_empty(), "ожидался непустой ответ");
}

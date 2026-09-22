//! Перекодировка при загрузке через фасад: GGUF (Q8_0 / Q4_K_M) → SQ4 в
//! памяти, ответ на «столица Франции» — «Paris». `SYN_TRANSCODE_BIG=1`
//! добавляет NVFP4-бандл qwen3.8-27b → SQ4 (15 ГБ RAM, минуты).
//! Запуск: cargo test -p synaptix --release --test transcode_smoke -- --nocapture --test-threads=1

use std::path::PathBuf;

use synaptix::facade::llm::{load_llm_with_policy, GenerationOptions, LlmGeneration, QuantPolicy};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

fn opts(max_new: usize) -> GenerationOptions {
    GenerationOptions {
        max_new_tokens: max_new,
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
    }
}

fn answer(path: &PathBuf, policy: QuantPolicy) -> String {
    let (model, tokenizer) = load_llm_with_policy(path, policy, &Device::Cuda(0)).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let ids = tokenizer.encode("The capital of France is").expect("encode");
    let mut g = LlmGeneration::new(&model, opts(6));
    g.set_stop_tokens(tokenizer.eos_ids().to_vec());
    let mut raw = String::new();
    g.generate_streaming(&ids, &tokenizer, |_id, delta| {
        raw.push_str(delta);
        true
    })
    .expect("generate");
    raw
}

#[test]
fn gguf_transcoded_to_sq4_answers() {
    if synaptix_core::device::cuda::get(0).is_err() {
        return;
    }
    let d = home().join("Storage/syn_models/gguf-tests");
    let mut models = vec![d.join("Qwen3-0.6B-Q8_0.gguf"), d.join("Llama-3.2-1B-Instruct-Q4_K_M.gguf")];
    if std::env::var("SYN_TRANSCODE_BIG").is_ok() {
        models.push(home().join("Storage/syn_models/qwen3.8-27b.syn"));
    }
    for path in models.into_iter().filter(|p| p.is_file()) {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        // SQ8 из Q8_0/Q4_K_M почти без потерь — ответ обязан сохраниться;
        // SQ4 на 0,6B — справочно: четырёх бит такой модели мало и в
        // llama.cpp (Q4_K_M у 0.6B заметно хуже), поэтому проверяем его на
        // 1B и больше.
        for bits in [8u8, 4] {
            let mut policy = QuantPolicy::balance();
            policy.weights_storage = DType::Sq { bits };
            policy.lm_head_storage = DType::Sq { bits: bits.max(6) };
            policy.embed_storage = DType::F16;
            policy.transcode = true;
            let t0 = std::time::Instant::now();
            let raw = answer(&path, policy);
            eprintln!("[{name}] sq{bits}: {:?} → {raw:?}", t0.elapsed());
            let strict = bits == 8 || !name.contains("0.6B");
            assert!(!strict || raw.contains("Paris"), "{}: sq{bits} → {raw:?}", path.display());
        }
    }
}

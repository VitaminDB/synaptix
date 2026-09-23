//! Прямая загрузка `.gguf` через фасад и жадная генерация на эталонных
//! файлах llama.cpp (пропуск, если файлов нет):
//! `~/Storage/syn_models/gguf-tests/*.gguf` (Qwen3-0.6B Q8_0, Gemma-3-1B Q8_0,
//! Llama-3.2-1B Q4_K_M) и Qwen2 7B Q4_0 из Downloads. Ответ на «столица
//! Франции» обязан содержать «Париж»/«Paris»; для сырого промпта «The capital
//! of France is» ожидается « Paris», как у llama-completion.
//!
//! Запуск: cargo test -p synaptix --release --test gguf_facade_smoke -- --nocapture --test-threads=1

use std::path::PathBuf;

use synaptix::facade::llm::{load_llm_with_policy, GenerationOptions, LlmGeneration, Message, QuantPolicy};
use synaptix_core::device::Device;

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_default())
}

fn models() -> Vec<PathBuf> {
    let d = home().join("Storage/syn_models/gguf-tests");
    let mut v = vec![
        d.join("Qwen3-0.6B-Q8_0.gguf"),
        d.join("gemma-3-1b-it-Q8_0.gguf"),
        d.join("Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        home().join("Storage/vitamindb/vitamindb/Downloads/ggml-model-Q4_0.gguf"),
    ];
    if let Ok(only) = std::env::var("SYN_GGUF_ONLY") {
        v.retain(|p| p.to_string_lossy().contains(&only));
    }
    v.into_iter().filter(|p| p.is_file()).collect()
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

#[test]
fn gguf_models_answer_capital_of_france() {
    let ms = models();
    if ms.is_empty() {
        eprintln!("нет эталонных GGUF — пропуск");
        return;
    }
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("нет CUDA — пропуск");
        return;
    }
    for path in ms {
        let t0 = std::time::Instant::now();
        let (model, tokenizer) = load_llm_with_policy(&path, QuantPolicy::quality(), &Device::Cuda(0))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        eprintln!("[{}] загрузка {:?}", path.file_name().unwrap().to_string_lossy(), t0.elapsed());

        // 1. Сырой промпт, как в llama-completion: ожидаем « Paris».
        let ids = tokenizer.encode("The capital of France is").expect("encode");
        eprintln!("  ids {ids:?}");
        let lax = std::env::var("SYN_GGUF_LAX").is_ok();
        let mut g = LlmGeneration::new(&model, opts(6));
        g.set_stop_tokens(tokenizer.eos_ids().to_vec());
        let mut raw = String::new();
        g.generate_streaming(&ids, &tokenizer, |_id, delta| {
            raw.push_str(delta);
            true
        })
        .expect("generate raw");
        eprintln!("  raw → {raw:?}");
        assert!(lax || raw.contains("Paris"), "{}: сырой промпт → {raw:?}", path.display());

        // 2. Чат-шаблон модели.
        let msgs = [Message::user("What is the capital of France? Answer in one word.")];
        let prompt = tokenizer.apply_chat_template_ex_tools(&msgs, true, false, None).expect("chat template");
        let ids = tokenizer.encode(&prompt).expect("encode");
        eprintln!("  eos {:?}\n  prompt {prompt:?}", tokenizer.eos_ids());
        let mut g = LlmGeneration::new(&model, opts(48));
        g.set_stop_tokens(tokenizer.eos_ids().to_vec());
        let mut out = String::new();
        g.generate_streaming(&ids, &tokenizer, |_id, delta| {
            out.push_str(delta);
            true
        })
        .expect("generate chat");
        eprintln!("  chat → {out:?}");
        assert!(lax || out.contains("Paris") || out.contains("Париж"), "{}: чат → {out:?}", path.display());

        // 3. Шаблон с инструментами — так зовёт чат synthos. Llama-3.x пишет
        // их через `tojson(indent=4)` (кварг), раньше рендер падал.
        let tool = serde_json::json!({
            "type": "function",
            "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {}}}
        });
        let with_tools = tokenizer
            .apply_chat_template_ex_tools(&msgs, true, false, Some(&[tool]))
            .unwrap_or_else(|e| panic!("{}: шаблон с tools: {e}", path.display()));
        // Gemma-3 инструменты в шаблоне не выводит — требуем только рендер.
        eprintln!("  tools в промпте: {}", with_tools.contains("get_weather"));
    }
}

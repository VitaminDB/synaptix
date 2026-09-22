//! `probe_llm_answer <model> [prompt]` — жадные 6 токенов на сыром промпте.
use synaptix::facade::llm::{load_llm_with_policy, GenerationOptions, LlmGeneration, QuantPolicy};
use synaptix_core::device::Device;

fn main() {
    synaptix::init().unwrap();
    let path = std::env::args().nth(1).unwrap();
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "The capital of France is".into());
    let (model, tokenizer) = load_llm_with_policy(std::path::Path::new(&path), QuantPolicy::quality(), &Device::Cuda(0)).unwrap();
    let ids = tokenizer.encode(&prompt).unwrap();
    let opts = GenerationOptions { max_new_tokens: 6, max_seq_len: 2048, temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0, repeat_penalty: 1.0, repeat_last_n: 0, presence_penalty: 0.0, frequency_penalty: 0.0 };
    let mut g = LlmGeneration::new(&model, opts);
    g.set_stop_tokens(tokenizer.eos_ids().to_vec());
    let mut out = String::new();
    g.generate_streaming(&ids, &tokenizer, |_id, d| { out.push_str(d); true }).unwrap();
    println!("ids {ids:?}\nraw → {out:?}");
}

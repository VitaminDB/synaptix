//! `lora_gen <model.syn> <adapter_dir> [вопрос]` — ответ без адаптера, с ним
//! (наложение при генерации) и снова без: скорость и текст.
use std::time::Instant;

use synaptix::facade::llm::{load_llm_with_policy, GenerationOptions, LlmGeneration, Message, QuantPolicy};
use synaptix_core::device::Device;

fn main() {
    synaptix::init().unwrap();
    let a: Vec<String> = std::env::args().collect();
    let q = a.get(3).cloned().unwrap_or_else(|| "Объясни в двух предложениях, что такое фотосинтез.".into());
    let (model, tok) = load_llm_with_policy(std::path::Path::new(&a[1]), QuantPolicy::balance(), &Device::Cuda(0)).unwrap();
    let prompt = tok.apply_chat_template_ex_tools(&[Message::user(q)], true, false, None).unwrap();
    let ids = tok.encode(&prompt).unwrap();
    let opts = GenerationOptions { max_new_tokens: 120, max_seq_len: 2048, temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0, repeat_penalty: 1.0, repeat_last_n: 0, presence_penalty: 0.0, frequency_penalty: 0.0 };
    let run = |label: &str| {
        let mut g = LlmGeneration::new(&model, opts.clone());
        g.set_stop_tokens(tok.eos_ids().to_vec());
        let mut out = String::new();
        let mut n = 0usize;
        let t0 = Instant::now();
        g.generate_streaming(&ids, &tok, |_i, d| {
            out.push_str(d);
            n += 1;
            true
        })
        .unwrap();
        let dt = t0.elapsed().as_secs_f64();
        println!("── {label}: {n} ток за {dt:.2} с ({:.1} ток/с)\n{}\n", n as f64 / dt, out.trim());
    };
    run("без адаптера");
    let info = model.load_lora("test", std::path::Path::new(&a[2])).unwrap();
    println!("загружен: {} матриц, r={}, {:.1} МБ", info.modules, info.rank, info.bytes as f64 / 1e6);
    run("с адаптером");
    model.set_lora_strength("test", 0.5).unwrap();
    run("сила 0.5");
    model.set_lora_enabled("test", false).unwrap();
    run("адаптер выключен");
}

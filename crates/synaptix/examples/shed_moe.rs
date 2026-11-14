//! `shed_moe <model> [prompt]` — MoE целиком на карте → `shed_blocks()` (так
//! служба отвечает на нехватку видеопамяти) → эксперты в RAM лениво из
//! бандла; генерация до и после должна совпасть, после — на той же карте.
use synaptix::facade::llm::{load_llm, GenerationOptions, LlmGeneration};
use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;

fn main() {
    synaptix::init().unwrap();
    let path = std::env::args().nth(1).unwrap();
    let prompt = std::env::args().nth(2).unwrap_or_else(|| "<|im_start|>user\nОбъясни, как работает хеш-таблица с открытой адресацией.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".into());
    let (model, tok) = load_llm(std::path::Path::new(&path), Device::Cuda(0), PrecisionConfig::nvfp4(), Some(4096)).unwrap();
    let ids = tok.encode(&prompt).unwrap();
    let run = |what: &str| -> String {
        let opts = GenerationOptions { max_new_tokens: 200, max_seq_len: 4096, temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0, seed: 0, repeat_penalty: 1.0, repeat_last_n: 0, presence_penalty: 0.0, frequency_penalty: 0.0 };
        let mut g = LlmGeneration::new(&model, opts);
        g.set_stop_tokens(tok.eos_ids().to_vec());
        let mut out = String::new();
        let mut n = 0usize;
        let t = std::time::Instant::now();
        g.generate_streaming(&ids, &tok, |_id, d| { out.push_str(d); n += 1; true }).unwrap();
        let free = synaptix_core::device::cuda::mem_info(0).map(|m| m.0 as f64 / 1e9).unwrap_or(0.0);
        println!("{what}: {n} токенов за {:.2} с ({:.1} ток/с), свободно VRAM {free:.1} ГБ", t.elapsed().as_secs_f64(), n as f64 / t.elapsed().as_secs_f64());
        out
    };
    let a = run("на карте");
    let _ = run("на карте, повтор");
    println!("shed_blocks → {}", model.shed_blocks());
    let b = run("эксперты в RAM");
    let _ = run("эксперты в RAM, повтор");
    let same = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    println!("совпадает первых символов: {same} из {} / {}", a.chars().count(), b.chars().count());
    println!("--- до:\n{}\n--- после:\n{}", a.chars().take(300).collect::<String>(), b.chars().take(300).collect::<String>());
}

//! Стоимость сэмплера на словаре LFM3 (128000) с параметрами по умолчанию модели.
use synaptix_core::{device::Device, tensor::Tensor};
use synaptix_llm_common::generate::{GenerationConfig, TokenSampler};

fn main() {
    let cfg = GenerationConfig { temperature: 0.1, top_k: 50, repetition_penalty: 1.1, max_new_tokens: 1, ..Default::default() };
    let prompt: Vec<u32> = (0..300).map(|i| (i * 7919 % 128000) as u32).collect();
    let mut s = TokenSampler::new(&cfg, &prompt);
    let v: Vec<f32> = (0..128000).map(|i| ((i as f32 * 0.013).sin()) * 8.0).collect();
    let t = Tensor::from_vec(v, vec![1usize, 128000], Device::Cpu).unwrap();
    for _ in 0..5 {
        s.sample(&t).unwrap();
    }
    let n = 100;
    let t0 = std::time::Instant::now();
    for _ in 0..n {
        s.sample(&t).unwrap();
    }
    println!("сэмплер: {:.3} мс на токен", t0.elapsed().as_secs_f64() * 1e3 / n as f64);
}

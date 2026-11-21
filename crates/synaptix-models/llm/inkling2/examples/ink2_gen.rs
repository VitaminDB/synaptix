//! Жадная генерация без шаблона: `cargo run --example ink2_gen -- <hf-dir> "<prompt>" [n]`.
//! Печатает токены промпта, топ-5 логитов первого шага и продолжение.

use std::time::Instant;

use synaptix_core::device::Device;
use synaptix_llm_inkling2::{Inkling2Model, Inkling2Weights};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let prompt = args.get(2).map(String::as_str).unwrap_or("The capital of France is");
    let n_new: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let t0 = Instant::now();
    let w = Inkling2Weights::open(dir).expect("open");
    let model = Inkling2Model::load(&w, dev).expect("load");
    eprintln!("загрузка {:.1} с", t0.elapsed().as_secs_f32());
    let tok = HfTokenizer::from_bytes(&w.tokenizer_json).expect("tokenizer");
    let ids = tok.encode(prompt, false).expect("encode").ids;
    println!("ids {:?}", ids);
    let mut cache = model.make_cache(ids.len() + n_new + 8).expect("cache");
    let t1 = Instant::now();
    let logits = model.forward(&ids, &mut cache, false).expect("forward");
    let lv = logits.to_device(Device::Cpu).unwrap().reshape(vec![logits.dims()[1]]).unwrap().to_vec1::<f32>().unwrap();
    eprintln!("префилл {} ток. {:.2} с", ids.len(), t1.elapsed().as_secs_f32());
    let mut order: Vec<usize> = (0..lv.len()).collect();
    order.sort_by(|a, b| lv[*b].total_cmp(&lv[*a]));
    for &i in order.iter().take(5) {
        println!("top {i} {:.4} {:?}", lv[i], tok.decode(&[i as u32], false).unwrap_or_default());
    }
    let mut out = vec![order[0] as u32];
    let t2 = Instant::now();
    while out.len() < n_new {
        let l = model.decode(*out.last().unwrap(), &mut cache).expect("step");
        let v = l.to_device(Device::Cpu).unwrap().reshape(vec![l.dims()[1]]).unwrap().to_vec1::<f32>().unwrap();
        let best = v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        out.push(best as u32);
    }
    eprintln!("декод {} ток. {:.2} ток/с", out.len() - 1, (out.len() - 1) as f32 / t2.elapsed().as_secs_f32());
    eprint!("{}", synaptix_llm_inkling2::kern::prof_report());
    println!("gen ids {:?}", out);
    println!("text {:?}", tok.decode(&out, false).unwrap_or_default());
}

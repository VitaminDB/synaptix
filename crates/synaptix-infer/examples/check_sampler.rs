//! Проверка КОРРЕКТНОСТИ нового сэмплера: старый алгоритм (полная сортировка)
//! vs новый (select_nth + finite-filter) — одинаковый ли набор выживших токенов.
use synaptix_infer::sampling::top_k::TopKProcessor;
use synaptix_infer::sampling::top_p::TopPProcessor;
use synaptix_infer::sampling::{LogitProcessor, ProcessorContext};

fn old_topk(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() { return; }
    let mut idx: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for i in k..idx.len() { logits[idx[i].0] = f32::NEG_INFINITY; }
}
fn old_topp(logits: &mut [f32], p: f32) {
    if p >= 1.0 { return; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for q in probs.iter_mut() { *q /= sum; }
    let mut idx: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32; let mut cut = idx.len();
    for (pos, (_, pr)) in idx.iter().enumerate() { if cum > p { cut = pos; break; } cum += pr; }
    for i in cut..idx.len() { logits[idx[i].0] = f32::NEG_INFINITY; }
}

fn main() {
    let v = 248320usize;
    let ctx = ProcessorContext { input_ids: Vec::new(), step: 0, batch_idx: 0 };
    let mut mism = 0;
    for trial in 0..5u64 {
        let mut s = 0x9E3779B97F4A7C15u64 ^ (trial.wrapping_mul(0xD1B54A32D192ED03));
        // распределение с разной «остротой»
        let scale = 3.0 + trial as f32 * 2.0;
        let base: Vec<f32> = (0..v).map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) * scale - scale*0.5
        }).collect();
        let mut a = base.clone(); old_topk(&mut a, 40); old_topp(&mut a, 0.9);
        let mut b = base.clone();
        TopKProcessor{k:40}.process(&mut b, &ctx).unwrap();
        TopPProcessor{p:0.9}.process(&mut b, &ctx).unwrap();
        let fa: std::collections::BTreeSet<usize> = a.iter().enumerate().filter(|(_,x)|x.is_finite()).map(|(i,_)|i).collect();
        let fb: std::collections::BTreeSet<usize> = b.iter().enumerate().filter(|(_,x)|x.is_finite()).map(|(i,_)|i).collect();
        let eq = fa == fb;
        if !eq { mism += 1; }
        println!("trial {trial} scale={scale}: старый_выжило={} новый_выжило={} наборы_совпали={}", fa.len(), fb.len(), eq);
    }
    println!("\n{}", if mism==0 {"✅ КОРРЕКТНО: наборы идентичны"} else {"❌ БАГ: наборы РАЗЛИЧАЮТСЯ — сэмплер сломан"});
}

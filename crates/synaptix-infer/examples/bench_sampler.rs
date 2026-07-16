//! Стоимость сэмплинга на токен (CPU), словарь 248320: НОВЫЙ путь
//! (top_k=40 select_nth + top_p=0.9 filter) vs СТАРАЯ полная сортировка словаря.
//! cargo run --profile fast-release -p synaptix-infer --example bench_sampler

use std::time::Instant;

use synaptix_infer::sampling::top_k::TopKProcessor;
use synaptix_infer::sampling::top_p::TopPProcessor;
use synaptix_infer::sampling::{LogitProcessor, ProcessorContext};

fn main() {
    let v = 248320usize;
    let ctx = ProcessorContext { input_ids: Vec::new(), step: 0, batch_idx: 0 };
    let mut s = 0x9E3779B97F4A7C15u64;
    let base: Vec<f32> = (0..v)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / u32::MAX as f32) * 12.0 - 6.0
        })
        .collect();
    let iters = 200u32;

    // НОВЫЙ: top_k=40 (select_nth) → top_p=0.9 (сортирует только 40).
    let t = Instant::now();
    let mut acc = 0.0f32;
    for _ in 0..iters {
        let mut l = base.clone();
        TopKProcessor { k: 40 }.process(&mut l, &ctx).unwrap();
        TopPProcessor { p: 0.9 }.process(&mut l, &ctx).unwrap();
        acc += l.iter().filter(|x| x.is_finite()).count() as f32;
    }
    let new_us = t.elapsed().as_secs_f64() * 1e6 / iters as f64;

    // СТАРЫЙ (симуляция): полная сортировка всего словаря (как было в top_p при top_k=0).
    let t2 = Instant::now();
    for _ in 0..iters {
        let mut idx: Vec<(usize, f32)> = base.iter().copied().enumerate().collect();
        idx.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        std::hint::black_box(&idx);
    }
    let old_us = t2.elapsed().as_secs_f64() * 1e6 / iters as f64;

    println!("словарь V={v}, iters={iters}");
    println!("НОВЫЙ (top_k=40 select_nth + top_p filter): {new_us:.1} us/токен");
    println!("СТАРЫЙ (полная сортировка V):               {old_us:.1} us/токен");
    println!("ускорение сэмплинга: {:.1}×  (экономия {:.0} us/токен)", old_us / new_us.max(1e-6), old_us - new_us);
    std::hint::black_box(acc);
}

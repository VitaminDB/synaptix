//! Квантифицирует стоимость per-token детокенизации в chat decode-цикле:
//! СТАРАЯ (decode всей последовательности каждый токен, O(N²)) vs НОВАЯ
//! (vLLM-окно [prefix..], O(N)). Запуск:
//!   cargo run --profile fast-release --example bench_chat_detok --features cuda -- "/path/model.syn"
#![cfg(feature = "cuda")]

use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let path = std::env::args().nth(1).expect("usage: bench_chat_detok MODEL.syn");
    let prec = PrecisionConfig::from_preset("nvfp4").unwrap();
    let prec = PrecisionConfig { compute: synaptix_core::dtype::DType::F16, ..prec };
    let pipe = HybridPipeline::load_with_precision(std::path::Path::new(&path), Device::Cuda(0), prec, Some(4096))
        .expect("load");
    // Длинный текст → реальные id (эмулируем 400-токенный ответ).
    let text = "Париж — столица Франции. ".repeat(60);
    let ids = pipe.encode(&text).expect("encode");
    let n = ids.len().min(400);
    let ids = &ids[..n];
    println!("симулируем стриминг {n} токенов");

    // СТАРАЯ: decode(&ids[..i]) каждый токен.
    let t0 = std::time::Instant::now();
    let mut emitted = 0usize;
    let mut old_out = String::new();
    for i in 1..=n {
        let full = pipe.decode(&ids[..i]).unwrap();
        if full.len() > emitted && full.is_char_boundary(emitted) && !full[emitted..].ends_with('\u{FFFD}') {
            old_out.push_str(&full[emitted..]);
            emitted = full.len();
        }
    }
    let old_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // НОВАЯ: окно [prefix_offset..].
    let t1 = std::time::Instant::now();
    let (mut prefix_offset, mut read_offset) = (0usize, 0usize);
    let mut new_out = String::new();
    for i in 1..=n {
        let cur = &ids[..i];
        let prefix_text = if prefix_offset >= read_offset {
            String::new()
        } else {
            pipe.decode(&cur[prefix_offset..read_offset]).unwrap()
        };
        let new_text = pipe.decode(&cur[prefix_offset..]).unwrap();
        if new_text.len() > prefix_text.len()
            && new_text.is_char_boundary(prefix_text.len())
            && !new_text.ends_with('\u{FFFD}')
        {
            new_out.push_str(&new_text[prefix_text.len()..]);
            prefix_offset = read_offset;
            read_offset = i;
        }
    }
    let new_ms = t1.elapsed().as_secs_f64() * 1000.0;

    println!("СТАРАЯ (O(N²) full-decode): {old_ms:.1} ms ({:.3} ms/токен)", old_ms / n as f64);
    println!("НОВАЯ (vLLM-окно):          {new_ms:.1} ms ({:.3} ms/токен)", new_ms / n as f64);
    println!("ускорение детокенизации: {:.1}×", old_ms / new_ms.max(1e-6));
    println!("вывод идентичен: {}", old_out == new_out);
}

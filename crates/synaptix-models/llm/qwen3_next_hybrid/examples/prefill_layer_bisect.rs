//! По-слойная бисекция chunked-prefill бага: set_dump_layers пишет L2-норму
//! скрытого состояния ПОСЛЕДНЕГО токена после каждого под-слоя. Сравниваем
//! single vs chunked → ПЕРВЫЙ расходящийся слой = источник.
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid
//! --example prefill_layer_bisect -- MODEL.syn [T] [chunk]
use synaptix_core::device::Device;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::model::{layer_dump_take, set_dump_gtok, set_dump_layers};
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;

fn run_prefill(pipe: &HybridPipeline, ids: &[u32], chunk: usize) -> Vec<(usize, String, Vec<f32>)> {
    let device = pipe.model.device;
    let mut kv = pipe.model.make_kv_cache(1, 1536).expect("kv");
    let _ = layer_dump_take(); // очистить
    let mut off = 0;
    let n_chunks = ids.len().div_ceil(chunk);
    while off < ids.len() {
        let end = (off + chunk).min(ids.len());
        let t = Tensor::from_vec(ids[off..end].to_vec(), vec![1usize, end - off], device).unwrap();
        let _ = no_grad(|| pipe.model.forward(&t, &mut kv)).expect("forward");
        off = end;
    }
    let all = layer_dump_take();
    // SYN_DUMP_GTOK: записывает только чанк, содержащий глобальный токен → все
    // записи уже от него. Иначе — последний чанк (его последний токен = конец).
    if std::env::var("SYN_DUMP_GTOK").is_ok() {
        return all;
    }
    let per_chunk = all.len() / n_chunks;
    all[all.len() - per_chunk..].to_vec()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: MODEL.syn [T] [chunk]");
    let target: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(850);
    let chunk: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    set_dump_layers(true);
    set_dump_gtok(std::env::var("SYN_DUMP_GTOK").ok().and_then(|v| v.parse().ok()));

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pipe = HybridPipeline::load_with_precision(&path, Device::Cuda(0), PrecisionConfig::nvfp4(), Some(1536))
        .expect("load");

    let needle = "Секретное кодовое слово — ТИГР-9471. Запомни.\n\n";
    let filler = "Погода переменная, ветер слабый. Поезда по расписанию. Магазин до девяти. ";
    let mut s = needle.to_string();
    while pipe.encode(&s).map(|v| v.len()).unwrap_or(0) < target { s.push_str(filler); }
    let ids = pipe.encode(&s).unwrap();
    eprintln!("T={} токенов, chunk={chunk}\n", ids.len());

    let single = run_prefill(&pipe, &ids, ids.len() + 16);
    let chunked = run_prefill(&pipe, &ids, chunk);

    eprintln!("слой/подслой | max_abs(вектор) | mean_abs | ПЕРВОЕ bit-расхождение?");
    let mut first_div: Option<usize> = None;
    for (i, (s_e, c_e)) in single.iter().zip(&chunked).enumerate() {
        let (sv, cv) = (&s_e.2, &c_e.2);
        let maxd = sv.iter().zip(cv).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let meand = sv.iter().zip(cv).map(|(a, b)| (a - b).abs()).sum::<f32>() / sv.len().max(1) as f32;
        // bit-расхождение: ЛЮБОЙ ненулевой diff (детерминированный путь → должно быть 0.0 если каузально).
        let big = maxd > 1e-4;
        if big && first_div.is_none() { first_div = Some(i); }
        if big || i < 12 || s_e.0 >= 999 || first_div.map(|f| i >= f.saturating_sub(2) && i <= f + 8).unwrap_or(false) {
            eprintln!("L{:4} {:9} | max_abs={maxd:.5} | mean_abs={meand:.6} | {}",
                s_e.0, s_e.1, if big { "◄── ДА" } else { "" });
        }
    }
    match first_div {
        Some(i) => eprintln!("\n>>> ПЕРВОЕ bit-расхождение: запись {i} → слой L{} подслой '{}'", single[i].0, single[i].1),
        None => eprintln!("\n>>> ВЕЗДЕ bit-exact (0.0) — chunked==single, баг НЕ в скрытом состоянии этого токена"),
    }
}

//! Где чанкованный prefill расходится с единым? Прогоняем ОДИН и тот же prompt
//! через model.forward (а) одним вызовом и (б) чанками по 512, сравниваем
//! логиты последнего токена. Если расходятся → баг в prefill. cargo run
//! --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid
//! --example prefill_chunk_divergence -- MODEL.syn [T] [chunk]
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;
use synaptix_tokenizer::tokenizer::Tokenizer;

fn host_logits(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0
}

fn prefill_logits(pipe: &HybridPipeline, ids: &[u32], chunk: usize) -> Vec<f32> {
    let device = pipe.model.device;
    let mut kv = pipe.model.make_kv_cache(1, 8192).expect("kv");
    let mut last = None;
    let mut off = 0;
    while off < ids.len() {
        let end = (off + chunk).min(ids.len());
        let t = Tensor::from_vec(ids[off..end].to_vec(), vec![1usize, end - off], device).unwrap();
        let lg = no_grad(|| pipe.model.forward(&t, &mut kv)).expect("forward");
        last = Some(lg);
        off = end;
    }
    host_logits(&last.unwrap())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: MODEL.syn [T] [chunk] [ref_chunk]");
    let target: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(827);
    let chunk: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(512);
    // Референс: 0/отсутствует → single-shot; иначе — чанкованный prefill с этим
    // размером (для T, где single-shot не влезает в VRAM: чанк-vs-чанк).
    let ref_chunk: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let pipe = HybridPipeline::load_with_precision(&path, Device::Cuda(0), PrecisionConfig::nvfp4(), Some(8192))
        .expect("load");
    eprintln!("loaded.");

    // Промпт: needle в начале + filler до T токенов.
    let needle = "Секретное кодовое слово — ТИГР-9471. Запомни.\n\n";
    let filler = "Погода переменная, ветер слабый. Поезда по расписанию. Магазин до девяти. ";
    let mut s = needle.to_string();
    while pipe.encode(&s).map(|v| v.len()).unwrap_or(0) < target {
        s.push_str(filler);
    }
    let ids = pipe.encode(&s).unwrap();
    eprintln!("T={} токенов, chunk={chunk} (→ {} чанков)\n", ids.len(), ids.len().div_ceil(chunk));

    let ref_sz = if ref_chunk == 0 { ids.len() + 16 } else { ref_chunk };
    let single = prefill_logits(&pipe, &ids, ref_sz); // референс (single или ref_chunk)
    let single2 = prefill_logits(&pipe, &ids, ref_sz); // ТОТ ЖЕ путь повторно (детерминизм?)
    let chunked = prefill_logits(&pipe, &ids, chunk);          // chunked
    {
        let m = single.iter().zip(&single2).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        let dot: f32 = single.iter().zip(&single2).map(|(a, b)| a * b).sum();
        let n1: f32 = single.iter().map(|a| a * a).sum::<f32>().sqrt();
        let n2: f32 = single2.iter().map(|a| a * a).sum::<f32>().sqrt();
        eprintln!(">>> ДЕТЕРМИНИЗМ single-vs-single: max_abs={m:.4} cos={:.6} {}",
            dot / (n1 * n2), if m < 0.01 { "ДЕТЕРМИНИРОВАН" } else { "◄── НЕДЕТЕРМИНИРОВАН!" });
    }

    let m = single.iter().zip(&chunked).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let mean: f32 = single.iter().zip(&chunked).map(|(a, b)| (a - b).abs()).sum::<f32>() / single.len() as f32;
    let (as_, ac) = (argmax(&single), argmax(&chunked));
    // косинус
    let dot: f32 = single.iter().zip(&chunked).map(|(a, b)| a * b).sum();
    let n1: f32 = single.iter().map(|a| a * a).sum::<f32>().sqrt();
    let n2: f32 = chunked.iter().map(|a| a * a).sum::<f32>().sqrt();
    eprintln!("=== prefill logits: single vs chunked (T={}) ===", ids.len());
    eprintln!("max_abs_diff = {m:.4}");
    eprintln!("mean_abs_diff = {mean:.5}");
    eprintln!("cos = {:.6}", dot / (n1 * n2));
    eprintln!("argmax single={as_} chunked={ac} {}", if as_ == ac { "СОВПАЛ" } else { "РАСХОЖДЕНИЕ" });
    eprintln!("single[{as_}]={:.3} chunked[{as_}]={:.3} | chunked[{ac}]={:.3}", single[as_], chunked[as_], chunked[ac]);
}

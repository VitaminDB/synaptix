//! Сравнение prefill+decode по форматам на Qwen3-1.7B (dense HF-дир, влезает во все
//! форматы в отличие от 27B). prefill = forward(prompt) мульти-итер min на разных
//! длинах (включая невыровненную 350 — проверка NVFP4 M-паддинга). decode = generate
//! greedy, decode_ms → tok/s.
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3
//!   --example format_perf -- models/Qwen/Qwen3-1.7B [f16 bf16 nvfp4 mxfp8]
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::GenerationConfig;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;

fn precision_for(name: &str) -> Option<PrecisionConfig> {
    match name {
        "f16" => Some(PrecisionConfig::dense(DType::F16)),
        "bf16" => Some(PrecisionConfig::dense(DType::BF16)),
        "nvfp4" => Some(PrecisionConfig::nvfp4()),
        "mxfp8" => Some(PrecisionConfig::mxfp8()),
        _ => None,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .expect("usage: format_perf MODEL_DIR [f16 bf16 nvfp4 mxfp8]");
    let fmts: Vec<String> = {
        let v: Vec<String> = args.collect();
        if v.is_empty() {
            vec!["f16".into(), "bf16".into(), "nvfp4".into(), "mxfp8".into()]
        } else {
            v
        }
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let dev = Device::Cuda(0);
    let lens: Vec<usize> = vec![128, 350, 512, 1024];
    let cap = lens.iter().max().copied().unwrap_or(2048) + 128;
    let stream = synaptix_core::device::cuda::default_stream(0).unwrap();

    for fmt in &fmts {
        let Some(prec) = precision_for(fmt) else {
            eprintln!("неизвестный формат {fmt}, пропуск");
            continue;
        };
        let t0 = std::time::Instant::now();
        let pipe = match Qwen3Pipeline::load_with_precision(&path, dev, prec, Some(cap)) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("\n### {fmt}: ЗАГРУЗКА FAIL: {e} ###");
                continue;
            }
        };
        eprintln!(
            "\n### {} | compute={:?} attn_w={:?} mlp_w={:?} | загрузка {:.1}с ###",
            fmt.to_uppercase(),
            prec.compute,
            prec.attn_w,
            prec.mlp_w,
            t0.elapsed().as_secs_f32()
        );

        // ── prefill ──
        eprintln!("  len(ток) | prefill min/med ms |   tok/s");
        for &n in &lens {
            let ids: Vec<u32> = (0..n).map(|i| ((i * 7 + 13) % 200 + 5) as u32).collect();
            let t = Tensor::from_vec(ids, vec![1usize, n], dev).unwrap();
            for _ in 0..3 {
                let mut kv = pipe.model.make_kv_cache(1, cap).unwrap();
                let _ = no_grad(|| pipe.model.forward(&t, &mut kv)).unwrap();
            }
            stream.synchronize().unwrap();
            let iters = if n <= 512 { 10 } else { 6 };
            let mut times = Vec::with_capacity(iters);
            for _ in 0..iters {
                let mut kv = pipe.model.make_kv_cache(1, cap).unwrap();
                let t1 = std::time::Instant::now();
                let _ = no_grad(|| pipe.model.forward(&t, &mut kv)).unwrap();
                stream.synchronize().unwrap();
                times.push(t1.elapsed().as_secs_f64() * 1000.0);
            }
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ms = times[0];
            let med = times[times.len() / 2];
            let tps = n as f64 / (ms / 1000.0);
            eprintln!("  {n:8} | {ms:7.1} / {med:7.1}   | {tps:8.0}");
        }

        // ── decode (greedy, decode_ms → tok/s) ──
        let prompt_ids: Vec<u32> = (0..32).map(|i| ((i * 7 + 13) % 200 + 5) as u32).collect();
        let cfg = GenerationConfig {
            max_new_tokens: 64,
            temperature: 0.0,
            max_seq: Some(cap),
            ..Default::default()
        };
        let dtps = |run: &dyn Fn() -> Option<(Vec<u32>, synaptix_llm_common::GenerationStats)>| -> f64 {
            let mut best = 0.0_f64;
            for _ in 0..2 {
                let Some((out, stats)) = run() else { return -1.0 };
                let new = out.len().saturating_sub(prompt_ids.len());
                if stats.decode_ms > 0 && new > 1 {
                    let v = (new as f64 - 1.0) / (stats.decode_ms as f64 / 1000.0);
                    if v > best {
                        best = v;
                    }
                }
            }
            best
        };
        let d_plain = dtps(&|| no_grad(|| pipe.generate(&prompt_ids, cfg.clone())).ok());
        let d_graph = dtps(&|| no_grad(|| pipe.generate_with_graph(&prompt_ids, cfg.clone())).ok());
        let g = if d_graph < 0.0 {
            "N/A".to_string()
        } else {
            format!("{d_graph:.1}")
        };
        eprintln!("  decode greedy | plain {d_plain:6.1} | +graph {g} tok/s");
        drop(pipe);
    }
}

//! End-to-end проверка MXFP8-GEMM в chunked-prefill: теряет ли контекст (как был
//! NVFP4 wrong-row баг) или держит. Qwen3-1.7B (dense, влезает в VRAM в MXFP8, в
//! отличие от 27B). needle в начале промпта, вопрос в конце; chunked prefill
//! (batch=512, пишет KV через MXFP8 q/k/v-проекции) vs single. Если chunked recall
//! падает → MXFP8-GEMM строко-некорректен; если держит → M-разброс benign (fp-order).
//! SYN_NEEDLE_QUANT=nvfp4|mxfp8 (дефолт mxfp8).
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3
//!   --example mxfp8_needle -- models/Qwen/Qwen3-1.7B [T]
use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;
use synaptix_llm_common::GenerationConfig;
use synaptix_tokenizer::tokenizer::Tokenizer;

fn wrap(user: &str) -> String {
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
}

fn build(pipe: &Qwen3Pipeline, target: usize) -> Vec<u32> {
    let needle = "Секретное кодовое слово — ТИГР-9471. Запомни — я спрошу позже.\n\n";
    let filler = "Погода переменная, ветер слабый. Поезда по расписанию. Магазин до девяти. Кот спит на окне. ";
    let tail = "\n\nКакое секретное кодовое слово я назвал в самом начале? Назови точно.";
    let mut body = String::new();
    loop {
        let p = wrap(&format!("{needle}{body}{tail}"));
        let n = pipe.encode(&p).map(|v| v.len()).unwrap_or(0);
        if n >= target { return pipe.encode(&p).unwrap(); }
        body.push_str(filler);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: mxfp8_needle MODEL_DIR [T...]");
    let targets: Vec<usize> = {
        let v: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() { vec![827, 1643] } else { v }
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let precision = match std::env::var("SYN_NEEDLE_QUANT").as_deref() {
        Ok("nvfp4") => PrecisionConfig::nvfp4(),
        _ => PrecisionConfig::mxfp8(),
    };
    eprintln!("quant: {:?} (attn_w), compute {:?}", precision.attn_w, precision.compute);
    let pipe = Qwen3Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(4096))
        .expect("load");
    let im_end = pipe.tokenizer.token_to_id("<|im_end|>");
    let cfg = GenerationConfig {
        max_new_tokens: 64, temperature: 0.0, top_k: 0, top_p: 1.0, min_p: 0.0,
        repetition_penalty: 1.0, repeat_last_n: 0, presence_penalty: 0.0,
        frequency_penalty: 0.0, seed: 1, eos_token_id: None,
        eos_token_ids: im_end.into_iter().collect(), max_seq: Some(4096), prefill_batch: 512,
    };
    let run = |ids: &[u32], batch: usize| -> (bool, String) {
        let mut c = cfg.clone(); c.prefill_batch = batch;
        let mut kv = pipe.model.make_kv_cache(1, 4096).expect("kv");
        let mut sink = |_t: u32| true;
        let (out, _) = pipe.generate_with_graph_resume(&mut kv, ids, c, &mut sink).expect("gen");
        let t = pipe.decode(&out).unwrap_or_default();
        (t.contains("ТИГР") || t.contains("9471"), t.replace('\n', " ⏎ ").trim().to_string())
    };
    for &target in &targets {
        let ids = build(&pipe, target);
        eprintln!("\n════ T={} токенов ════", ids.len());
        let (rc, tc) = run(&ids, 512);
        eprintln!("  chunked batch=512 recall={rc} | {}", tc.chars().take(110).collect::<String>());
        let (rs, ts) = run(&ids, ids.len() + 16);
        eprintln!("  single  batch=ALL recall={rs} | {}", ts.chars().take(110).collect::<String>());
        eprintln!("  → {}", if rc == rs { "chunked==single ✅ (MXFP8-GEMM держит контекст)" }
            else { "❌ chunked≠single — MXFP8-GEMM теряет контекст (wrong-row баг)" });
    }
}

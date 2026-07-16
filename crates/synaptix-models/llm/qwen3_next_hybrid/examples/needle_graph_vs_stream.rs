//! Диагностика бага «теряет контекст»: needle-in-haystack на РЕАЛЬНОЙ 27B-hybrid
//! модели через ДВА пути decode — graph (что юзает чат) и streaming. Greedy
//! (детерминированно) → токены обоих путей сравнимы напрямую.
//!
//! Промпт RAW (без chat-template/thinking): факт в начале → длинный filler →
//! вопрос в конце. Если модель не вспоминает факт на длинном контексте, но
//! вспоминает на коротком — баг в prefill-длине. Если graph ≠ streaming — баг
//! в graph-handoff.
//!
//! cargo run --profile fast-release --features cuda -p synaptix-llm-qwen3-next-hybrid \
//!   --example needle_graph_vs_stream -- "models/qwen3.6 27B.syn" [target_tokens]
use synaptix_core::device::Device;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;
use synaptix_llm_common::GenerationConfig;
use synaptix_tokenizer::tokenizer::Tokenizer;

// Qwen chat-формат с ОТКЛЮЧЁННЫМ thinking (пустой <think></think> → модель
// отвечает сразу, не тратит бюджет на рассуждения).
fn wrap_chat(user: &str) -> String {
    format!(
        "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
}

fn build_prompt(pipe: &HybridPipeline, target_tokens: usize) -> (String, Vec<u32>) {
    let needle = "Секретное кодовое слово — ТИГР-9471.";
    let head = format!("{needle} Запомни его — я спрошу позже. Ниже длинный нейтральный текст, его можно игнорировать.\n\n");
    let filler_unit = "Погода сегодня переменная, местами облачно, ветер слабый. Поезда ходят по расписанию. \
        Магазин открыт с девяти утра до девяти вечера. В библиотеке появились новые книги. \
        На рынке свежие овощи и фрукты. Кот спит на подоконнике под лучами солнца. ";
    let tail = "\n\nКакое секретное кодовое слово я назвал в самом начале? Назови его точно.";

    let mut body = String::new();
    loop {
        let user = format!("{head}{body}{tail}");
        let candidate = wrap_chat(&user);
        let n = pipe.encode(&candidate).map(|v| v.len()).unwrap_or(0);
        if n >= target_tokens {
            let ids = pipe.encode(&candidate).unwrap();
            return (candidate, ids);
        }
        body.push_str(filler_unit);
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: needle_graph_vs_stream MODEL.syn [lengths...]");
    let targets: Vec<usize> = {
        let v: Vec<usize> = args.filter_map(|s| s.parse().ok()).collect();
        if v.is_empty() { vec![120, 800, 1600, 3000, 6000] } else { v }
    };

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    // SYN_NEEDLE_QUANT=mxfp8 → веса MXFP8 (проверка chunked-prefill recall на MXFP8-GEMM);
    // иначе дефолт NVFP4.
    let precision = match std::env::var("SYN_NEEDLE_QUANT").as_deref() {
        Ok("mxfp8") | Ok("fp8") => PrecisionConfig::mxfp8(),
        _ => PrecisionConfig::nvfp4(),
    };
    eprintln!("quant preset: {:?} (attn_w)", precision.attn_w);
    eprintln!("loading {path} (nvfp4, compute={:?})...", precision.compute);
    let t0 = std::time::Instant::now();
    let pipe = HybridPipeline::load_with_precision(&path, device, precision, Some(8192))
        .expect("load");
    eprintln!("loaded in {:.1}s\n", t0.elapsed().as_secs_f32());

    let im_end = pipe.tokenizer.token_to_id("<|im_end|>");
    let eos_ids: Vec<u32> = im_end.into_iter().collect();
    let base_cfg = GenerationConfig {
        max_new_tokens: 80,
        temperature: 0.0, // greedy → детерминизм
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        seed: 1,
        eos_token_id: None,
        eos_token_ids: eos_ids,
        max_seq: Some(8192),
        prefill_batch: 512, // как в чате
    };

    eprintln!("needle = «ТИГР-9471» в самом начале промпта; вопрос в конце; thinking OFF.");
    eprintln!("ЭКСПЕРИМЕНТ: чанкованный prefill (batch=512) vs ОДНИМ чанком (batch=весь промпт).\n");
    let run = |ids: &[u32], batch: usize| -> (bool, String) {
        let mut cfg = base_cfg.clone();
        cfg.prefill_batch = batch;
        let mut kv = pipe.model.make_kv_cache(1, 8192).expect("kv");
        let mut sink = |_t: u32| true;
        let (out, _st) = pipe
            .generate_with_graph_resume(&mut kv, ids, cfg, &mut sink)
            .expect("graph gen");
        let txt = pipe.decode(&out).unwrap_or_default();
        (txt.contains("ТИГР") || txt.contains("9471"), txt.replace('\n', " ⏎ ").trim().to_string())
    };

    // STREAMING-путь (generic model.forward для decode, БЕЗ graph/forward_decode_dev).
    let run_stream = |ids: &[u32], batch: usize| -> (bool, String) {
        let mut cfg = base_cfg.clone();
        cfg.prefill_batch = batch;
        let mut kv = pipe.model.make_kv_cache(1, 8192).expect("kv");
        let mut sink = |_t: u32| true;
        let (out, _st) = pipe
            .generate_streaming_resume(&mut kv, ids, cfg, &mut sink)
            .expect("stream gen");
        let txt = pipe.decode(&out).unwrap_or_default();
        (txt.contains("ТИГР") || txt.contains("9471"), txt.replace('\n', " ⏎ ").trim().to_string())
    };

    for &target in &targets {
        let (_prompt, ids) = build_prompt(&pipe, target);
        let n = ids.len();
        eprintln!("════════ T={n} токенов ════════");
        let (r0, t0) = run(&ids, 0); // чат-дефолт: prefill_batch=0 → single-shot (мой фикс)
        eprintln!("  GRAPH  batch=0(chat) recall={r0} | {}", trunc(&t0, 140));
        let (rc, tc) = run(&ids, 512);
        eprintln!("  GRAPH  batch=512 recall={rc} | {}", trunc(&tc, 140));
        let (rsc, tsc) = run_stream(&ids, 512);
        eprintln!("  STREAM batch=512 recall={rsc} | {}", trunc(&tsc, 140));
        let (rs, ts) = run(&ids, n + 16);
        eprintln!("  GRAPH  batch=ALL recall={rs} | {}", trunc(&ts, 140));
        eprintln!();
    }
}

fn trunc(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { s.chars().take(n).collect::<String>() + "…" }
}

//! Гейтед-проверка «чат-LLM на малой карте»: путь чата synthos
//! (`optimal_profile` → `load_llm_with_policy` → подгонка блоков под контекст
//! → `generate_streaming_cached` с префикс-KV-сессией) при балласте в VRAM.
//! Фоновый поток ловит минимум свободной памяти.
//!
//! ```sh
//! LLM_MODEL=~/Storage/syn_models/qwen3.8-27b.syn LLM_VRAM_GB=7 \
//!   cargo test -p synaptix --release --test llm_low_vram -- --nocapture
//! ```
//! env: `LLM_VRAM_GB` — сколько оставить модели (без него — вся карта),
//! `LLM_PROMPT_TOKENS` (≈2000) — длина промпта, `LLM_NEW_TOKENS` (64),
//! `LLM_TURNS` (2) — ходов в одной сессии (второй — префикс-KV).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, set_dflash_enabled, set_graph_decode_enabled, set_mtp_enabled,
    GenerationOptions, Llm, LlmGeneration, Message,
};
use synaptix_core::device::cuda::{mem_info, synchronize_all};
use synaptix_core::device::Device;
use synaptix_core::memory::cuda_pool::hard_trim_all_pools_device;

fn free_vram() -> usize {
    mem_info(0).map(|(f, _)| f).unwrap_or(0)
}

fn gb(b: usize) -> f64 {
    b as f64 / 1e9
}

/// Как `fit_blocks_for_context` чата synthos (та же арифметика и запас
/// «холодного» хода 3 ГБ под кэши ядер и активации): не хватает — блоки
/// уезжают на хост, есть запас в два блока и больше — возвращаются.
fn fit_blocks(model: &Llm, tokens: usize) {
    let (Some((block_bytes, total)), Some(resident)) = (model.block_offload_shape(), model.resident_blocks()) else {
        return;
    };
    if block_bytes == 0 || total == 0 {
        return;
    }
    let _ = synchronize_all(0);
    let _ = hard_trim_all_pools_device(0);
    let free = free_vram();
    let need = tokens * model.kv_bytes_per_token() + model.kv_fixed_bytes(tokens) + (3072usize << 20);
    let want = if need > free {
        let missing = need - free;
        resident.saturating_sub(missing.div_ceil(block_bytes) + usize::from(missing >= block_bytes))
    } else {
        let spare = (free - need) / block_bytes;
        if spare < 2 {
            resident
        } else if resident + spare + 1 >= total {
            total
        } else {
            resident + spare
        }
    };
    let got = if want != resident { model.set_block_residency(want).unwrap_or(resident) } else { resident };
    let _ = hard_trim_all_pools_device(0);
    eprintln!(
        "[low_vram] блоки {resident} → {got} из {total} (свободно было {:.2} ГБ, ходу нужно {:.2}; теперь свободно {:.2})",
        gb(free),
        gb(need),
        gb(free_vram())
    );
}

#[test]
fn chat_llm_on_small_card() {
    let Ok(path) = std::env::var("LLM_MODEL") else {
        eprintln!("LLM_MODEL не задан — пропуск");
        return;
    };
    let path = std::path::PathBuf::from(path);
    let env_num = |k: &str, d: usize| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
    let prompt_tokens = env_num("LLM_PROMPT_TOKENS", 2000);
    let new_tokens = env_num("LLM_NEW_TOKENS", 64);
    let turns = env_num("LLM_TURNS", 2).max(1);

    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();

    let _ballast = std::env::var("LLM_VRAM_GB").ok().and_then(|v| v.parse::<f64>().ok()).map(|keep_gb| {
        let keep = (keep_gb * 1e9) as usize;
        let bytes = free_vram().saturating_sub(keep).max(1);
        eprintln!("[low_vram] балласт {:.1} ГБ → модели {:.1} ГБ", gb(bytes), gb(keep));
        synaptix_core::tensor::Tensor::zeros((bytes,), synaptix_core::dtype::DType::U8, Device::Cuda(0))
            .expect("балласт")
    });
    let avail0 = free_vram();
    let min_free = Arc::new(AtomicUsize::new(avail0));
    let stop = Arc::new(AtomicBool::new(false));
    let watcher = {
        let (min_free, stop) = (min_free.clone(), stop.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                min_free.fetch_min(free_vram(), Ordering::Relaxed);
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
        })
    };

    let opt = optimal_profile(&path);
    set_graph_decode_enabled(opt.graph_decode && std::env::var_os("LLM_NO_GRAPH").is_none());
    set_mtp_enabled(opt.speculation);
    set_dflash_enabled(opt.speculation);
    let t0 = Instant::now();
    let (model, tokenizer) = load_llm_with_policy(&path, opt.policy, &Device::Cuda(0)).expect("загрузка");
    let _ = synchronize_all(0);
    let _ = hard_trim_all_pools_device(0);
    eprintln!(
        "[low_vram] {} загружена за {:.1} с: блоков на карте {:?} из {:?}, занято {:.2} ГБ, свободно {:.2} ГБ",
        path.display(),
        t0.elapsed().as_secs_f64(),
        model.resident_blocks(),
        model.block_offload_shape().map(|s| s.1),
        gb(avail0.saturating_sub(free_vram())),
        gb(free_vram())
    );

    // Длинный промпт: абзац-«документ» до нужной длины и вопрос в конце.
    let para = "The river town kept its old stone bridge, its market on Saturdays and a library \
                that never closed its reading room. ";
    let mut doc = String::new();
    while tokenizer.encode(&doc).map(|v| v.len()).unwrap_or(0) < prompt_tokens {
        for _ in 0..20 {
            doc.push_str(para);
        }
    }
    let questions = ["What is the capital of France? Answer with one word.", "And of Italy? One word."];
    let mut history = vec![Message::system("You are a concise assistant."), Message::user(format!("{doc}\n\n{}", questions[0]))];

    let ctx = prompt_tokens + turns * (new_tokens + 64) + 256;
    fit_blocks(&model, ctx);
    let mut session = model.new_kv_session(ctx, new_tokens).expect("сессия");
    let mut answers = Vec::new();
    for turn in 0..turns {
        let prompt = tokenizer.apply_chat_template_ex_tools(&history, true, false, None).expect("шаблон");
        let ids = tokenizer.encode(&prompt).expect("encode");
        let mut generation = LlmGeneration::new(
            &model,
            GenerationOptions {
                max_new_tokens: new_tokens,
                max_seq_len: ctx,
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 0,
                repeat_penalty: 1.0,
                repeat_last_n: 0,
                presence_penalty: 0.0,
                frequency_penalty: 0.0,
            },
        );
        generation.set_stop_tokens(tokenizer.eos_ids().to_vec());
        let mut out = String::new();
        let mut first: Option<f64> = None;
        let mut n = 0usize;
        let t = Instant::now();
        let reused = match session.as_mut() {
            Some(s) => generation
                .generate_streaming_cached(s, &ids, &tokenizer, |_, d| {
                    first.get_or_insert(t.elapsed().as_secs_f64());
                    n += 1;
                    out.push_str(d);
                    true
                })
                .expect("генерация"),
            None => {
                generation.generate_streaming(&ids, &tokenizer, |_, d| {
                    first.get_or_insert(t.elapsed().as_secs_f64());
                    n += 1;
                    out.push_str(d);
                    true
                })
                .expect("генерация");
                0
            }
        };
        let total = t.elapsed().as_secs_f64();
        let ttft = first.unwrap_or(total);
        eprintln!(
            "[low_vram] ход {}: промпт {} ток. (из кэша {reused}), первый токен {ttft:.1} с, декод {:.1} ток/с; ответ: {:?}",
            turn + 1,
            ids.len(),
            (n.saturating_sub(1)) as f64 / (total - ttft).max(1e-3),
            out.trim()
        );
        history.push(Message::assistant(out.trim()));
        answers.push(out);
        if let Some(q) = questions.get(turn + 1) {
            history.push(Message::user(*q));
        } else {
            break;
        }
    }
    stop.store(true, Ordering::Relaxed);
    let _ = watcher.join();
    let low = min_free.load(Ordering::Relaxed);
    eprintln!(
        "[low_vram] минимум свободной VRAM {:.2} ГБ (пик модели ≈ {:.2} ГБ из {:.2})",
        gb(low),
        gb(avail0.saturating_sub(low)),
        gb(avail0)
    );
    assert!(answers[0].to_lowercase().contains("paris"), "ответ без Paris: {}", answers[0]);
}

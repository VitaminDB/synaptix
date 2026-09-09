//! Замер скорости префилла Gemma-4: длинный промпт, разные чанки префилла.
//!
//! Запуск:
//!   SYN_GEMMA4_BUNDLE=…/gemma-4-26b-a4b-it.syn \
//!   SYN_BENCH_PROMPT=4096 SYN_BENCH_CHUNKS=0,512,2048 [SYN_MOE_PROFILE=1] \
//!   cargo test -p synaptix --release --test gemma4_prefill_bench -- --nocapture --test-threads=1

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_common::GenerationConfig;
use synaptix_llm_gemma4::Gemma4Pipeline;

fn bundle() -> Option<String> {
    std::env::var("SYN_GEMMA4_BUNDLE").ok()
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// Промпт из повторяющихся абзацев примерно на `want` токенов.
fn prompt_ids(pipe: &Gemma4Pipeline, want: usize) -> Vec<u32> {
    use synaptix_tokenizer::Tokenizer;
    let para = "Пункт {i}. Регламент обслуживания предписывает проверять журналы, \
                обновлять сертификаты, вести учёт заявок в общем реестре и раз в \
                неделю сверять остатки на складе с данными учётной системы.\n";
    let mut text = String::from("<bos><|turn>user\n");
    let mut i = 0usize;
    loop {
        text.push_str(&para.replace("{i}", &i.to_string()));
        i += 1;
        if i % 16 == 0 {
            let n = pipe.tokenizer.encode(&text, false).expect("encode").ids.len();
            if n + 64 >= want {
                break;
            }
        }
    }
    text.push_str("Сколько всего пунктов в регламенте выше?<turn|>\n<|turn>model\n<|channel>thought\n<channel|>");
    pipe.tokenizer.encode(&text, false).expect("encode").ids
}

#[test]
fn prefill_bench() {
    let Some(path) = bundle() else { return };
    synaptix::init().expect("init");
    let precision = PrecisionConfig {
        compute: DType::BF16,
        attn_w: DType::MXFP8,
        mlp_w: DType::NVFP4,
        lm_head: DType::NVFP4,
        embed: DType::BF16,
        kv: DType::BF16,
    };
    let want = env_usize("SYN_BENCH_PROMPT", 4096);
    let pipe = Gemma4Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(want * 2 + 512))
        .expect("load gemma4");
    let ids = prompt_ids(&pipe, want);
    let max_seq = ids.len() + 64;
    eprintln!("[prefill] промпт {} ток.", ids.len());

    let chunks: Vec<usize> = std::env::var("SYN_BENCH_CHUNKS")
        .unwrap_or_else(|_| "0".into())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let cfg = |chunk: usize| GenerationConfig {
        max_new_tokens: 1,
        temperature: 0.0,
        max_seq: Some(max_seq),
        prefill_batch: chunk,
        ..Default::default()
    };
    // Прогрев: скретчи ядер, дескрипторы, перемешанные копии.
    let short: Vec<u32> = ids[..ids.len().min(512)].to_vec();
    let _ = pipe.generate(&short, cfg(0)).expect("warmup");
    let _ = synaptix_llm_common::profile::report();

    let profile = std::env::var("SYN_PREFILL_PROF").is_ok();
    let repeats = env_usize("SYN_BENCH_REPEAT", 1);
    for chunk in chunks {
        for _ in 0..repeats {
            if profile {
                synaptix_llm_common::model::set_decode_prof(true);
            }
            let (out, stats) = pipe.generate(&ids, cfg(chunk)).expect("generate");
            if profile {
                synaptix_llm_common::model::set_decode_prof(false);
            }
            let free = synaptix_core::device::cuda::mem_info(0).map(|(f, _)| f / (1 << 20)).unwrap_or(0);
            eprintln!(
                "[prefill] chunk={chunk}: {} ток. за {} мс → {:.0} ток/с; первый токен {:?}; VRAM свободно {} МБ",
                stats.prompt_tokens,
                stats.prefill_ms,
                stats.prompt_tokens as f64 * 1000.0 / stats.prefill_ms.max(1) as f64,
                pipe.decode(&out).unwrap_or_default(),
                free,
            );
            if profile {
                eprintln!("{}", synaptix_llm_common::model::decode_prof_report_and_clear());
            }
            let moe = synaptix_llm_common::profile::report();
            if !moe.trim().is_empty() {
                eprintln!("=== MoE stages ===\n{moe}");
            }
        }
    }
}

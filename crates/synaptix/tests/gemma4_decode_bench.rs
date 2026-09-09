//! Замер скорости декода Gemma-4 с разбивкой по стадиям.
//!
//! Запуск:
//!   SYN_GEMMA4_BUNDLE=…/gemma-4-26b-a4b-it.syn \
//!   cargo test -p synaptix --release --test gemma4_decode_bench -- --nocapture --test-threads=1

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_common::GenerationConfig;
use synaptix_llm_gemma4::Gemma4Pipeline;

fn bundle() -> Option<String> {
    std::env::var("SYN_GEMMA4_BUNDLE").ok()
}

fn prompt() -> String {
    "<bos><|turn>user\nНапиши подробный рассказ про кота, который научился \
     программировать. Не меньше двухсот слов.<turn|>\n\
     <|turn>model\n<|channel>thought\n<channel|>"
        .to_string()
}

/// Сколько байт весов декод читает на один токен: по ним считается потолок,
/// в который упирается память карты.
fn bytes_per_token(attn_w: DType, mlp_w: DType, head: DType) -> f64 {
    let bpp = |d: DType| match d {
        DType::NVFP4 => 0.5625,
        DType::MXFP8 => 1.0625,
        _ => 2.0,
    };
    let (h, layers) = (2816.0_f64, 30.0);
    // Внимание: 25 sliding-слоёв (голова 256) и 5 global (голова 512, V = K).
    let sliding = (4096.0 + 2048.0 + 2048.0 + 4096.0) * h * 25.0;
    let global = (8192.0 + 1024.0 + 8192.0) * h * 5.0;
    let attn = (sliding + global) * bpp(attn_w);
    // Плотный MLP каждого слоя плюс 8 активных экспертов из 128.
    let dense = 3.0 * 2112.0 * h * layers;
    let experts = 8.0 * (1408.0 * h + h * 704.0) * layers;
    let ffn = (dense + experts) * bpp(mlp_w);
    let lm_head = 262144.0 * h * bpp(head);
    attn + ffn + lm_head
}

fn run(name: &str, precision: PrecisionConfig, profile: bool) {
    let Some(path) = bundle() else { return };
    synaptix::init().expect("init");
    let pipe = Gemma4Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(2048))
        .expect("load gemma4");
    let ids = {
        use synaptix_tokenizer::Tokenizer;
        pipe.tokenizer.encode(&prompt(), false).expect("encode").ids
    };
    let cfg = |n: usize| GenerationConfig {
        max_new_tokens: n,
        temperature: 0.0,
        max_seq: Some(2048),
        prefill_batch: 256,
        ..Default::default()
    };
    // Прогрев: первый ход строит скретчи ядер и дескрипторы.
    let _ = pipe.generate(&ids, cfg(4)).expect("warmup");

    if profile {
        synaptix_llm_common::model::set_decode_prof(true);
    }
    let (out, stats) = pipe.generate(&ids, cfg(64)).expect("generate");
    if profile {
        synaptix_llm_common::model::set_decode_prof(false);
    }
    // Первый токен приходит из префилла — шагов декода на один меньше.
    let steps = out.len().saturating_sub(1).max(1);
    let per_tok = stats.decode_ms as f64 / steps as f64;
    let bpt = bytes_per_token(precision.attn_w, precision.mlp_w, precision.lm_head);
    eprintln!(
        "[{name}] {} ток.: decode {} мс → {:.2} мс/ток = {:.1} ток/с; \
         чтение весов {:.2} ГБ/ток → потолок при 800 ГБ/с {:.1} ток/с",
        out.len(),
        stats.decode_ms,
        per_tok,
        1000.0 / per_tok,
        bpt / 1e9,
        800e9 / bpt,
    );
    let text = pipe.decode(&out).unwrap_or_default();
    let head: String = text.chars().take(160).collect();
    eprintln!("[{name}] текст: {head:?}");
    if profile {
        eprintln!("{}", synaptix_llm_common::model::decode_prof_report_and_clear());
        let moe = synaptix_llm_common::profile::report();
        if !moe.trim().is_empty() {
            eprintln!("=== MoE stages ===\n{moe}");
        }
    }
}

#[test]
fn decode_bf16_attn() {
    run(
        "attn bf16, mlp nvfp4, head bf16",
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::BF16,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        std::env::var("SYN_DECODE_PROF").is_ok(),
    );
}

#[test]
fn decode_all_nvfp4() {
    run(
        "attn nvfp4, mlp nvfp4, head nvfp4",
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::NVFP4,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        std::env::var("SYN_DECODE_PROF").is_ok(),
    );
}

/// Разделяем вклад кванта внимания и кванта головы: у Gemma активации
/// «тяжёлые», и NVFP4 во внимании может стоить качества.
#[test]
fn decode_bf16_attn_nvfp4_head() {
    run(
        "attn bf16, mlp nvfp4, head nvfp4",
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::BF16,
            mlp_w: DType::NVFP4,
            lm_head: DType::NVFP4,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        false,
    );
}

#[test]
fn decode_nvfp4_attn_bf16_head() {
    run(
        "attn nvfp4, mlp nvfp4, head bf16",
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::NVFP4,
            mlp_w: DType::NVFP4,
            lm_head: DType::BF16,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        false,
    );
}

/// Ровно тот профиль, который synthos берёт по умолчанию.
#[test]
fn decode_optimal_profile() {
    let Some(path) = bundle() else { return };
    let profile = synaptix::facade::llm::optimal_profile(std::path::Path::new(&path));
    let precision = profile.policy.to_precision().expect("to_precision");
    eprintln!(
        "[optimal] compute={:?} attn={:?} mlp={:?} head={:?} embed={:?} kv={:?}",
        precision.compute,
        precision.attn_w,
        precision.mlp_w,
        precision.lm_head,
        precision.embed,
        precision.kv
    );
    run("optimal", precision, false);
}

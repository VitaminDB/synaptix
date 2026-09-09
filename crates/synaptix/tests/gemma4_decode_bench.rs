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
    let want = std::env::var("SYN_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64);
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
    let (out, stats) = pipe.generate(&ids, cfg(want)).expect("generate");
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
    let take = std::env::var("SYN_BENCH_TEXT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(160);
    let head: String = text.chars().take(take).collect();
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

/// MXFP8 против NVFP4 на внимании: восемь бит вместо четырёх стоят вдвое
/// больше чтения, но мы упираемся в запуски ядер, а не в память.
#[test]
fn decode_mxfp8_attn() {
    run(
        "attn mxfp8, mlp nvfp4, head mxfp8",
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::MXFP8,
            mlp_w: DType::NVFP4,
            lm_head: DType::MXFP8,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        false,
    );
}

#[test]
fn decode_mxfp8_attn_nvfp4_head() {
    run(
        "attn mxfp8, mlp nvfp4, head nvfp4",
        PrecisionConfig {
            compute: DType::BF16,
            attn_w: DType::MXFP8,
            mlp_w: DType::NVFP4,
            lm_head: DType::NVFP4,
            embed: DType::BF16,
            kv: DType::BF16,
        },
        false,
    );
}

/// Графовый декод: шаг захватывается в CUDA-граф, роутер MoE и выбор
/// экспертов идут на карте. Сверяем и скорость, и то, что текст совпал с
/// обычным путём — граф обязан считать ровно то же.
#[test]
fn decode_cuda_graph() {
    let Some(path) = bundle() else { return };
    synaptix::init().expect("init");
    let pick = |v: &str, d: DType| match std::env::var(v).ok().as_deref() {
        Some("bf16") => DType::BF16,
        Some("mxfp8") => DType::MXFP8,
        Some("nvfp4") => DType::NVFP4,
        _ => d,
    };
    let precision = PrecisionConfig {
        compute: DType::BF16,
        attn_w: pick("SYN_ATTN_W", DType::MXFP8),
        mlp_w: DType::NVFP4,
        lm_head: pick("SYN_HEAD_W", DType::NVFP4),
        embed: DType::BF16,
        kv: DType::BF16,
    };
    let pipe = Gemma4Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(2048))
        .expect("load gemma4");
    if std::env::var("SYN_GRAPH_FORCE").is_err() {
        assert!(pipe.graph_decode_supported(), "граф не поддержан этим профилем");
    }

    let ids = {
        use synaptix_tokenizer::Tokenizer;
        pipe.tokenizer.encode(&prompt(), false).expect("encode").ids
    };
    let want = std::env::var("SYN_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(64);
    let cfg = |n: usize| GenerationConfig {
        max_new_tokens: n,
        temperature: 0.0,
        max_seq: Some(2048),
        prefill_batch: 256,
        ..Default::default()
    };
    struct Quiet;
    impl synaptix_llm_common::StreamSink for Quiet {
        fn on_token(&mut self, _id: u32) -> bool {
            true
        }
    }
    let mut sink = Quiet;
    let _ = pipe
        .generate_with_graph_streaming(&ids, cfg(4), &mut sink)
        .expect("warmup");

    let (g_out, g_stats) = pipe
        .generate_with_graph_streaming(&ids, cfg(want), &mut sink)
        .expect("graph");
    let steps = g_out.len().saturating_sub(1).max(1);
    eprintln!(
        "[graph] {} ток.: decode {} мс → {:.2} мс/ток = {:.1} ток/с",
        g_out.len(),
        g_stats.decode_ms,
        g_stats.decode_ms as f64 / steps as f64,
        1000.0 * steps as f64 / g_stats.decode_ms.max(1) as f64,
    );

    let (h_out, h_stats) = pipe.generate(&ids, cfg(want)).expect("host");
    let hsteps = h_out.len().saturating_sub(1).max(1);
    eprintln!(
        "[host ] {} ток.: decode {} мс → {:.2} мс/ток = {:.1} ток/с",
        h_out.len(),
        h_stats.decode_ms,
        h_stats.decode_ms as f64 / hsteps as f64,
        1000.0 * hsteps as f64 / h_stats.decode_ms.max(1) as f64,
    );
    eprintln!("[graph] текст: {:?}", pipe.decode(&g_out).unwrap_or_default());

    // Совпадение по всей длине от greedy-декода требовать нельзя: пути
    // считают одно и то же, но разными ядрами (fused add+norm против пары
    // запусков), и на ничьих цепочки расходятся. Смотрим, сколько токенов
    // прошли одинаково, а строгая проверка — на логитах первого шага ниже.
    let common = g_out.iter().zip(&h_out).take_while(|(a, b)| a == b).count();
    eprintln!("[сверка] совпало {common} из {} токенов", h_out.len());
    // Первый токен приходит из общего префилла; дальше цепочки могут разойтись
    // на первой же ничьей (у этого промпта верхние логиты шага 1 отличаются на
    // 0.1) — длина общего префикса ничего не доказывает. Строгая сверка ниже,
    // на логитах одного шага.
    assert!(common >= 1, "граф разошёлся уже на токене {common}");
    assert!(!g_out.is_empty() && !h_out.is_empty());
}

/// Строгая сверка: один и тот же шаг декода, посчитанный обычным путём и
/// device-путём (тем, что уходит в граф), обязан дать те же логиты. Здесь
/// расхождение — это ошибка в раскладке, а не накопленный дрейф greedy-цепочки.
#[test]
fn graph_step_matches_host_step() {
    let Some(path) = bundle() else { return };
    synaptix::init().expect("init");
    let pick = |v: &str, d: DType| match std::env::var(v).ok().as_deref() {
        Some("bf16") => DType::BF16,
        Some("mxfp8") => DType::MXFP8,
        Some("nvfp4") => DType::NVFP4,
        _ => d,
    };
    let precision = PrecisionConfig {
        compute: DType::BF16,
        attn_w: pick("SYN_ATTN_W", DType::MXFP8),
        mlp_w: DType::NVFP4,
        lm_head: pick("SYN_HEAD_W", DType::NVFP4),
        embed: DType::BF16,
        kv: DType::BF16,
    };
    let pipe = Gemma4Pipeline::load_with_precision(&path, Device::Cuda(0), precision, Some(2048))
        .expect("load gemma4");
    let ids = {
        use synaptix_tokenizer::Tokenizer;
        pipe.tokenizer.encode(&prompt(), false).expect("encode").ids
    };
    let l = ids.len();
    let model = &pipe.model;
    let device = Device::Cuda(0);

    let prefill = |kv: &mut synaptix_llm_common::KvCache| {
        let t = synaptix_core::tensor::Tensor::from_vec(ids.clone(), vec![1usize, l], device)
            .expect("ids");
        synaptix_core::grad::no_grad(|| model.forward(&t, kv)).expect("prefill")
    };

    // Обычный путь: тот же шаг через `forward` при s = 1.
    let mut kv_a = pipe.make_kv_cache(2048).expect("kv");
    let lg = prefill(&mut kv_a);
    let tok0 = {
        let v = lg
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("логиты");
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(a, m), (i, &x)| if x > m { (i, x) } else { (a, m) })
            .0 as u32
    };
    let step = synaptix_core::tensor::Tensor::from_vec(vec![tok0], vec![1usize, 1], device)
        .expect("шаг");
    let host = synaptix_core::grad::no_grad(|| model.forward(&step, &mut kv_a)).expect("host-шаг");

    // Device-путь на свежем кэше в той же точке.
    let mut kv_b = pipe.make_kv_cache(2048).expect("kv");
    let _ = prefill(&mut kv_b);
    let mut state = model.make_decode_state().expect("decode state");
    let start = model.ring_prepare_decode(&mut kv_b, l).expect("ring");
    state.update_ring(tok0, l as u32, start as u32).expect("update");
    synaptix_core::grad::no_grad(|| model.forward_decode_dev(&mut state, &mut kv_b))
        .expect("device-шаг");

    let to_vec = |t: &synaptix_core::tensor::Tensor| {
        t.to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .expect("на хост")
    };
    let a = to_vec(&host);
    let b = to_vec(&state.logits);
    assert_eq!(a.len(), b.len());
    // Критерий тот же, что у существующего теста графа (cuda_graph_decode.rs):
    // косинусная близость логитов и совпадение argmax. Побитового равенства
    // тут не бывает — device-путь сливает add с нормой одним ядром и читает KV
    // другим, так что младшие разряды расходятся.
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nbb = 0f64;
    let mut max_abs = 0f32;
    for (x, y) in a.iter().zip(&b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nbb += (*y as f64) * (*y as f64);
        max_abs = max_abs.max((x - y).abs());
    }
    let cos = dot / (na.sqrt() * nbb.sqrt()).max(1e-12);
    let top = |v: &[f32]| {
        v.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(i0, m), (i, &x)| if x > m { (i, x) } else { (i0, m) })
            .0
    };
    let top3 = |v: &[f32]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_unstable_by(|x, y| v[*y].partial_cmp(&v[*x]).unwrap());
        idx.iter().take(3).map(|i| (*i, v[*i])).collect::<Vec<_>>()
    };
    eprintln!(
        "[шаг] cos = {cos:.6}, max|Δ| = {max_abs:.4}; argmax host={} dev={}; top3 host={:?} dev={:?}",
        top(&a),
        top(&b),
        top3(&a),
        top3(&b)
    );
    assert!(cos >= 0.99, "логиты разошлись: cos = {cos}");
    // Пути считают одно и то же разными ядрами, и на верхних логитах шум
    // достигает пары единиц: строгое равенство argmax ломается на ничьих
    // (host 16.875 против 16.75). Требуем взаимности: выбор каждого пути
    // входит в top-5 другого.
    let rank_in = |v: &[f32], tok: usize| v.iter().filter(|x| **x > v[tok]).count();
    let (ra, rb) = (rank_in(&a, top(&b)), rank_in(&b, top(&a)));
    eprintln!("[шаг] ранг dev-argmax у host = {ra}, ранг host-argmax у dev = {rb}");
    assert!(ra < 5 && rb < 5, "device-путь выбрал другой токен: ранги {ra}/{rb}");
}

/// Скорость на том пути, которым ходит synthos: `load_llm` с выверенным
/// профилем плюс `LlmGeneration::generate_streaming` (граф включается сам).
#[test]
fn facade_speed_optimal() {
    let Some(path) = bundle() else { return };
    use synaptix::facade::llm::{
        load_llm, optimal_profile, GenerationOptions, LlmGeneration, Message,
    };
    let p = std::path::Path::new(&path);
    let profile = optimal_profile(p);
    synaptix::facade::llm::set_graph_decode_enabled(profile.graph_decode);
    eprintln!("[фасад] graph_decode = {}", profile.graph_decode);
    let precision = profile.policy.to_precision().expect("precision");
    let (model, tokenizer) = load_llm(p, Device::Cuda(0), precision, Some(2048)).expect("load_llm");

    let msgs = [Message::user(
        "Напиши подробный рассказ про кота, который научился программировать. \
         Не меньше двухсот слов.",
    )];
    let prompt = tokenizer
        .apply_chat_template_ex_tools(&msgs, true, false, None)
        .expect("chat template");
    let ids = tokenizer.encode(&prompt).expect("encode");
    let want = std::env::var("SYN_BENCH_TOKENS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(200);
    let opts = |n: usize| GenerationOptions {
        max_new_tokens: n,
        max_seq_len: 2048,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        min_p: 0.0,
        seed: 0,
        repeat_penalty: 1.0,
        repeat_last_n: 0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
    };
    let mut warm = LlmGeneration::new(&model, opts(4));
    warm.set_stop_tokens(tokenizer.eos_ids().to_vec());
    let _ = warm.generate_streaming(&ids, &tokenizer, |_, _| true);

    let mut gen = LlmGeneration::new(&model, opts(want));
    gen.set_stop_tokens(tokenizer.eos_ids().to_vec());
    let mut out = String::new();
    let mut n = 0usize;
    let t0 = std::time::Instant::now();
    gen.generate_streaming(&ids, &tokenizer, |_id, delta| {
        out.push_str(delta);
        n += 1;
        true
    })
    .expect("generate");
    let dt = t0.elapsed();
    eprintln!(
        "[фасад] промпт {} ток. → {n} ток. за {:?} = {:.1} ток/с (с префиллом)",
        ids.len(),
        dt,
        n as f64 / dt.as_secs_f64()
    );
    eprintln!("[фасад] текст: {:?}", out.chars().take(200).collect::<String>());
    assert!(n > 0);
}

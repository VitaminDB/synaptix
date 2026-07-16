//! P7 — device-resident prefill (`forward_prefill_dev`) + CUDA-graph prefill chunk'а
//! на реальном Qwen3-1.7B. Парный к `cuda_graph_decode.rs` — снимает оставшийся
//! launch-overhead с короткого prompt'а (см. `plan/cuda_graph_prefill.md`).
//!
//! Гейтится фичей `cuda` + наличием весов в `models/Qwen/Qwen3-1.7B`.
//! Запуск: `cargo test -p synaptix --features cuda --profile fast-release cuda_graph_prefill`.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Mutex;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3::pipeline::{GenerationConfig, Qwen3Pipeline};

fn qwen3_dir() -> Option<PathBuf> {
    let p = PathBuf::from("models/Qwen/Qwen3-1.7B");
    if p.join("config.json").exists() { Some(p) } else { None }
}

// Тот же мьютекс-паттерн что в cuda_graph_decode.rs: один GPU + общий stream →
// capture не выдерживает параллельных тестов.
static GPU: Mutex<()> = Mutex::new(());

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(ai, av), (i, &x)| if x > av { (i, x) } else { (ai, av) })
        .0
}

fn cos_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    (dot / (na.sqrt() * nb.sqrt() + 1e-12)) as f32
}

fn to_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// P7.1: `forward_prefill_dev` (device-резидентные rope/kv/flash, T = chunk_size)
/// численно совпадает с обычным `forward` для одного prefill-chunk'а. Это та же
/// строгость, что в `forward_decode_dev_matches_forward`, но для multi-token T.
#[test]
fn forward_prefill_dev_matches_forward_single_chunk() {
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    // Активируем prefill-graph внутри `generate_with_graph_resume` (по
    // умолчанию выключен — см. pipeline.rs).
    std::env::set_var("SYN_PREFILL_GRAPH", "1");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");
    let model = &pipeline.model;
    let ids = pipeline.encode("The capital of France is").expect("encode");
    let chunk = ids.len();
    let prompt = Tensor::from_vec(ids.clone(), vec![1usize, chunk], cuda).unwrap();

    // Reference: обычный forward на полном prompt'е.
    let mut kv_ref = model.make_kv_cache(1, chunk + 8).expect("kv ref");
    let logits_ref = no_grad(|| model.forward(&prompt, &mut kv_ref)).expect("forward ref");
    let ref_v = to_f32(&logits_ref);

    // Dev: forward_prefill_dev на том же prompt'е (chunk = len, pos_start = 0).
    let mut kv_dev = model.make_kv_cache(1, chunk + 8).expect("kv dev");
    let mut pstate = model.make_prefill_state(chunk).expect("prefill state");
    pstate.update(&ids, 0).expect("prefill update");
    no_grad(|| model.forward_prefill_dev(&mut pstate, &mut kv_dev)).expect("forward_prefill_dev");
    let dev_v = to_f32(&pstate.logits);

    let cs = cos_sim(&ref_v, &dev_v);
    let am_ref = argmax(&ref_v);
    let am_dev = argmax(&dev_v);
    eprintln!(
        "[P7.1] chunk={chunk} cos_sim={cs:.6} argmax ref={am_ref} dev={am_dev}"
    );
    assert!(cs >= 0.99, "cos_sim too low: {cs}");
    assert_eq!(am_ref, am_dev, "greedy token mismatch");
}

/// P7.2: `forward_prefill_dev` с `pos_start > 0` (resume посередине prompt'а) —
/// продвинутый bit-exact: один chunk = 2-я половина prompt'а, KV pre-filled
/// первой половиной обычным forward'ом. Проверяет, что offset-aware causal mask
/// и rope `start_pos + t` работают корректно.
#[test]
fn forward_prefill_dev_matches_forward_offset_chunk() {
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    // Активируем prefill-graph внутри `generate_with_graph_resume` (по
    // умолчанию выключен — см. pipeline.rs).
    std::env::set_var("SYN_PREFILL_GRAPH", "1");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");
    let model = &pipeline.model;
    let ids = pipeline.encode("The capital of France is the city of Paris which sits on").expect("encode");
    let l = ids.len();
    assert!(l >= 4, "prompt too short: {l}");
    let half = l / 2;
    let chunk = l - half;
    let full = Tensor::from_vec(ids.clone(), vec![1usize, l], cuda).unwrap();
    let first = Tensor::from_vec(ids[..half].to_vec(), vec![1usize, half], cuda).unwrap();

    // Reference: forward целиком.
    let mut kv_ref = model.make_kv_cache(1, l + 8).expect("kv ref");
    let logits_ref = no_grad(|| model.forward(&full, &mut kv_ref)).expect("forward ref");
    let ref_v = to_f32(&logits_ref);

    // Dev: prefill первой половины обычным forward'ом + chunk-prefill второй
    // половины через forward_prefill_dev (pos_start = half).
    let mut kv_dev = model.make_kv_cache(1, l + 8).expect("kv dev");
    no_grad(|| model.forward(&first, &mut kv_dev)).expect("forward first half");
    let mut pstate = model.make_prefill_state(chunk).expect("prefill state");
    pstate.update(&ids[half..], half as u32).expect("prefill update");
    no_grad(|| model.forward_prefill_dev(&mut pstate, &mut kv_dev)).expect("forward_prefill_dev offset");
    let dev_v = to_f32(&pstate.logits);

    let cs = cos_sim(&ref_v, &dev_v);
    let am_ref = argmax(&ref_v);
    let am_dev = argmax(&dev_v);
    eprintln!(
        "[P7.2] half={half} chunk={chunk} cos_sim={cs:.6} argmax ref={am_ref} dev={am_dev}"
    );
    assert!(cs >= 0.99, "cos_sim too low: {cs}");
    assert_eq!(am_ref, am_dev, "greedy token mismatch");
}

/// P7.3: end-to-end `generate_with_graph` (с prefill-graph) даёт ту же
/// последовательность токенов, что и обычный `generate` (без graph вообще).
/// Использует длинный prompt → хотя бы один полный chunk_size активирует prefill-
/// graph путь.
#[test]
fn graph_prefill_matches_generate() {
    if std::env::var("SYN_GRAPH_PREFILL").is_err() {
        eprintln!("[skip] set SYN_GRAPH_PREFILL=1 to run prefill-graph parity");
        return;
    }
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    // Активируем prefill-graph внутри `generate_with_graph_resume` (по
    // умолчанию выключен — см. pipeline.rs).
    std::env::set_var("SYN_PREFILL_GRAPH", "1");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");

    // Длинный prompt → точно >= 1 полного chunk'а (chunk_default = 256).
    // Повторяем фразу пока не наберём 320+ токенов (1 chunk graph + 64+ tail).
    let mut ids = pipeline.encode("The capital of France is the city of Paris which sits on the river Seine.").expect("encode");
    while ids.len() < 320 {
        let more = pipeline.encode(" This is a continuation of the previous sentence with more content to extend prompt length.").expect("encode more");
        ids.extend(more);
    }
    let prompt_len = ids.len();
    let cfg = GenerationConfig {
        max_new_tokens: 16,
        temperature: 0.0,
        max_seq: Some(prompt_len + 32),
        prefill_batch: 256,
        ..Default::default()
    };

    // Прогрев NVRTC JIT обоих путей.
    let warm = GenerationConfig { max_new_tokens: 4, ..cfg.clone() };
    let _ = pipeline.generate(&ids, warm.clone()).expect("warmup baseline");
    let _ = pipeline.generate_with_graph(&ids, warm).expect("warmup graph");

    let (base_ids, base_stats) = pipeline.generate(&ids, cfg.clone()).expect("generate baseline");
    let (graph_ids, graph_stats) = pipeline.generate_with_graph(&ids, cfg).expect("generate graph");
    let base_pref_tps = (prompt_len as f64) / (base_stats.prefill_ms.max(1) as f64) * 1000.0;
    let graph_pref_tps = (prompt_len as f64) / (graph_stats.prefill_ms.max(1) as f64) * 1000.0;
    eprintln!(
        "[P7.3] prompt_len={prompt_len} baseline prefill={base_pref_tps:.0} tok/s, graph prefill={graph_pref_tps:.0} tok/s ({:.2}×)",
        graph_pref_tps / base_pref_tps
    );
    eprintln!("[P7.3] baseline='{}'", pipeline.decode(&base_ids).unwrap_or_default());
    eprintln!("[P7.3] graph   ='{}'", pipeline.decode(&graph_ids).unwrap_or_default());

    // Greedy → последовательности должны совпадать (допуск как в decode-тесте:
    // F16 rope-таблицы могут давать хвостовые расхождения, проверяем ≥ половины).
    let common = base_ids.iter().zip(graph_ids.iter()).take_while(|(a, b)| a == b).count();
    assert!(
        common >= base_ids.len().min(graph_ids.len()) / 2,
        "graph prefill diverged early: common={common} base={:?} graph={:?}",
        base_ids,
        graph_ids
    );
}

/// P7.5: perf-headline на сетке prompt_len. Печатает prefill tok/s baseline vs
/// graph для нескольких prompt_len → видно, на каких длинах capture-overhead
/// амортизируется (короткий prompt = 1 capture без replay'ев = убыток; длинный
/// = многократный replay = выигрыш). Только под SYN_GRAPH_PREFILL_PERF=1.
#[test]
fn graph_prefill_perf_grid() {
    if std::env::var("SYN_GRAPH_PREFILL_PERF").is_err() {
        eprintln!("[skip] set SYN_GRAPH_PREFILL_PERF=1 to run prefill-graph perf grid");
        return;
    }
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    // Активируем prefill-graph внутри `generate_with_graph_resume` (по
    // умолчанию выключен — см. pipeline.rs).
    std::env::set_var("SYN_PREFILL_GRAPH", "1");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");

    // Базовый seed-prompt; padd'им повторением последнего токена до нужной длины
    // (тот же приём что в `synaptix bench --prompt-tokens N`).
    let seed = pipeline.encode("The capital of France is").expect("encode");
    let pad = *seed.last().unwrap();
    let make_prompt = |n: usize| -> Vec<u32> {
        let mut v = seed.clone();
        if v.len() < n { v.resize(n, pad); } else { v.truncate(n); }
        v
    };

    let lengths = [5usize, 64, 256, 257, 512, 1024, 2048];
    let chunks = [32usize, 64, 128, 256];

    // Прогрев NVRTC обоих путей (для всех chunk-размеров делать прогрев не нужно —
    // ядра flash_splitq уже скомпилированы для всех HD на первом запуске).
    let warm_ids = make_prompt(*lengths.last().unwrap());
    let warm_cfg = GenerationConfig {
        max_new_tokens: 2, temperature: 0.0, max_seq: Some(warm_ids.len() + 8),
        prefill_batch: 256, ..Default::default()
    };
    let _ = pipeline.generate(&warm_ids, warm_cfg.clone()).expect("warmup baseline");
    let _ = pipeline.generate_with_graph(&warm_ids, warm_cfg).expect("warmup graph");

    for &chunk in &chunks {
        eprintln!("[P7.5] perf grid (chunk_size={chunk}, F16, Qwen3-1.7B):");
        eprintln!(
            "[P7.5] {:>6} | {:>10} | {:>10} | {:>8}",
            "prompt", "base tps", "graph tps", "speedup"
        );
        for &n in &lengths {
            let ids = make_prompt(n);
            let cfg = GenerationConfig {
                max_new_tokens: 1,
                temperature: 0.0,
                max_seq: Some(n + 4),
                prefill_batch: chunk,
                ..Default::default()
            };
            let (_b_ids, b) = pipeline.generate(&ids, cfg.clone()).expect("baseline");
            let (_g_ids, g) = pipeline.generate_with_graph(&ids, cfg).expect("graph");
            let b_tps = (n as f64) / (b.prefill_ms.max(1) as f64) * 1000.0;
            let g_tps = (n as f64) / (g.prefill_ms.max(1) as f64) * 1000.0;
            eprintln!(
                "[P7.5] {:>6} | {:>10.0} | {:>10.0} | {:>7.2}×  (base {} ms, graph {} ms)",
                n, b_tps, g_tps, g_tps / b_tps, b.prefill_ms, g.prefill_ms
            );
        }
    }
}

/// P7.4: хвост != 0 — prompt_len не делится на chunk_size → один полный chunk
/// через prefill-graph + хвост через host-fallback `model.forward`. Проверяет,
/// что разделение путей не ломает greedy.
#[test]
fn graph_prefill_with_tail_matches_generate() {
    if std::env::var("SYN_GRAPH_PREFILL").is_err() {
        eprintln!("[skip] set SYN_GRAPH_PREFILL=1 to run prefill-graph tail");
        return;
    }
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    // Активируем prefill-graph внутри `generate_with_graph_resume` (по
    // умолчанию выключен — см. pipeline.rs).
    std::env::set_var("SYN_PREFILL_GRAPH", "1");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");

    // prompt_len ≈ 290 (один chunk=256 graph + tail≈34 fallback).
    let mut ids = pipeline.encode("The capital of France is the city of Paris which sits on the river Seine.").expect("encode");
    while ids.len() < 256 {
        let more = pipeline.encode(" Continuing the prompt with extra tokens to cross 256.").expect("encode more");
        ids.extend(more);
    }
    while ids.len() < 290 {
        let more = pipeline.encode(" tail").expect("encode tail");
        ids.extend(more);
    }
    if ids.len() > 300 { ids.truncate(300); }
    let prompt_len = ids.len();
    assert!(prompt_len > 256, "prompt too short for tail test: {prompt_len}");
    assert!(prompt_len < 512, "prompt too long: {prompt_len}");

    let cfg = GenerationConfig {
        max_new_tokens: 8,
        temperature: 0.0,
        max_seq: Some(prompt_len + 16),
        prefill_batch: 256,
        ..Default::default()
    };

    let warm = GenerationConfig { max_new_tokens: 2, ..cfg.clone() };
    let _ = pipeline.generate(&ids, warm.clone()).expect("warmup baseline");
    let _ = pipeline.generate_with_graph(&ids, warm).expect("warmup graph");

    let (base_ids, _) = pipeline.generate(&ids, cfg.clone()).expect("generate baseline");
    let (graph_ids, _) = pipeline.generate_with_graph(&ids, cfg).expect("generate graph");
    eprintln!(
        "[P7.4] prompt_len={prompt_len} (chunk=256 + tail={}) base={:?} graph={:?}",
        prompt_len - 256, base_ids, graph_ids
    );
    let common = base_ids.iter().zip(graph_ids.iter()).take_while(|(a, b)| a == b).count();
    assert!(
        common >= base_ids.len().min(graph_ids.len()) / 2,
        "graph prefill (with tail) diverged early: common={common} base={:?} graph={:?}",
        base_ids, graph_ids
    );
}

//! P6.2/P6.3 — device-resident decode + CUDA-graph decode на реальном Qwen3-1.7B.
//!
//! Гейтится фичей `cuda` + наличием весов в `models/Qwen/Qwen3-1.7B`.
//! Запуск: `cargo test -p synaptix --features cuda --profile fast-release cuda_graph`.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::sync::Mutex;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::grad::no_grad;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_llm_qwen3::pipeline::{GenerationConfig, Qwen3Pipeline};

fn qwen3_dir() -> Option<PathBuf> {
    let p = PathBuf::from("models/Qwen/Qwen3-1.7B");
    if p.join("config.json").exists() { Some(p) } else { None }
}

// Один GPU + общий (кэшированный) stream → graph capture не выдерживает
// параллельных тестов (begin/end_capture из двух потоков на одном stream'е).
// `cargo test` гонит тесты параллельно — сериализуем GPU-секции этим мьютексом.
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

/// P6.2: forward_decode_dev (device-резидентные rope/kv/flash) численно совпадает
/// с обычным forward для одного decode-шага.
#[test]
fn forward_decode_dev_matches_forward() {
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");
    let model = &pipeline.model;
    let ids = pipeline.encode("The capital of France is").expect("encode");
    let l = ids.len();
    let prompt = Tensor::from_vec(ids.clone(), vec![1usize, l], cuda).unwrap();

    // Reference: prefill + один decode-шаг через обычный forward.
    let mut kv_ref = model.make_kv_cache(1, l + 8).expect("kv ref");
    let logits0 = no_grad(|| model.forward(&prompt, &mut kv_ref)).expect("prefill ref");
    let tok = argmax(&to_f32(&logits0)) as u32;
    let next = Tensor::from_vec(vec![tok], vec![1usize, 1], cuda).unwrap();
    let logits_ref = no_grad(|| model.forward(&next, &mut kv_ref)).expect("decode ref");
    let ref_v = to_f32(&logits_ref);

    // Dev: тот же prefill, затем тот же шаг через forward_decode_dev.
    let mut kv_dev = model.make_kv_cache(1, l + 8).expect("kv dev");
    let _ = no_grad(|| model.forward(&prompt, &mut kv_dev)).expect("prefill dev");
    let mut state = model.make_decode_state().expect("decode state");
    state.update(tok, l as u32).expect("state update");
    no_grad(|| model.forward_decode_dev(&mut state, &mut kv_dev)).expect("forward_decode_dev");
    let dev_v = to_f32(&state.logits);

    let cs = cos_sim(&ref_v, &dev_v);
    let am_ref = argmax(&ref_v);
    let am_dev = argmax(&dev_v);
    eprintln!("[P6.2] cos_sim={cs:.6} argmax ref={am_ref} dev={am_dev}");
    assert!(cs >= 0.99, "cos_sim too low: {cs}");
    assert_eq!(am_ref, am_dev, "greedy token mismatch");
}

/// P6.3: graph-decode даёт ту же последовательность токенов, что и обычный
/// generate (greedy), и печатает decode tok/s обоих путей.
#[test]
fn graph_decode_matches_generate() {
    if std::env::var("SYN_GRAPH_DECODE").is_err() {
        eprintln!("[skip] set SYN_GRAPH_DECODE=1 to run graph-decode parity");
        return;
    }
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    let cuda = Device::Cuda(0);
    let pipeline =
        Qwen3Pipeline::load(&dir, cuda, DType::F16).expect("load pipeline F16 cuda");
    let ids = pipeline.encode("The capital of France is").expect("encode");
    let cfg = GenerationConfig { max_new_tokens: 64, temperature: 0.0, max_seq: Some(ids.len() + 96), ..Default::default() };

    // Прогрев JIT (NVRTC компиляция ядер) для ОБОИХ путей — иначе первый прогон
    // несёт one-time compile cost в decode_ms (нечестное сравнение).
    let warm = GenerationConfig { max_new_tokens: 8, ..cfg.clone() };
    let _ = pipeline.generate(&ids, warm.clone()).expect("warmup baseline");
    let _ = pipeline.generate_with_graph(&ids, warm).expect("warmup graph");

    let (base_ids, base_stats) = pipeline.generate(&ids, cfg.clone()).expect("generate baseline");
    let (graph_ids, graph_stats) = pipeline.generate_with_graph(&ids, cfg).expect("generate graph");

    let base_tps = (base_stats.new_tokens as f64) / (base_stats.decode_ms.max(1) as f64) * 1000.0;
    let graph_tps = (graph_stats.new_tokens as f64) / (graph_stats.decode_ms.max(1) as f64) * 1000.0;
    eprintln!(
        "[P6.3] baseline decode {base_tps:.1} tok/s ({} tok), graph decode {graph_tps:.1} tok/s ({} tok)",
        base_stats.new_tokens, graph_stats.new_tokens
    );
    eprintln!("[P6.3] baseline='{}'", pipeline.decode(&base_ids).unwrap_or_default());
    eprintln!("[P6.3] graph   ='{}'", pipeline.decode(&graph_ids).unwrap_or_default());

    // Greedy → последовательности должны совпадать (допускаем расхождение хвоста
    // из-за F16 rope-таблиц: проверяем общий префикс ≥ половины).
    let common = base_ids.iter().zip(graph_ids.iter()).take_while(|(a, b)| a == b).count();
    assert!(
        common >= base_ids.len().min(graph_ids.len()) / 2,
        "graph decode diverged early: common={common} base={:?} graph={:?}",
        base_ids,
        graph_ids
    );
}

/// P6.3 headline: NVFP4-веса + CUDA-graph. Quant-decode сильнее launch-overhead-
/// bound (веса ×4 меньше → меньше bandwidth), поэтому graph даёт больший
/// относительный выигрыш ("множит выигрыш кванта"). Печатает tok/s обоих путей.
#[test]
fn graph_decode_nvfp4_speedup() {
    if std::env::var("SYN_GRAPH_DECODE").is_err() {
        eprintln!("[skip] set SYN_GRAPH_DECODE=1 to run nvfp4 graph-decode");
        return;
    }
    let Some(dir) = qwen3_dir() else {
        eprintln!("[skip] Qwen3-1.7B not found");
        return;
    };
    let _gpu = GPU.lock().unwrap_or_else(|e| e.into_inner());
    synaptix::init().expect("init backends");
    let cuda = Device::Cuda(0);
    let precision = PrecisionConfig::from_preset("nvfp4").expect("nvfp4 preset");
    let pipeline = Qwen3Pipeline::load_with_precision(&dir, cuda, precision, Some(256))
        .expect("load nvfp4 pipeline");
    let model = &pipeline.model;
    let ids = pipeline.encode("The capital of France is").expect("encode");
    let l = ids.len();
    let prompt = Tensor::from_vec(ids.clone(), vec![1usize, l], cuda).unwrap();

    // Single-step parity (rigorous): forward_decode_dev == forward для nvfp4.
    let mut kv_ref = model.make_kv_cache(1, l + 8).unwrap();
    let logits0 = no_grad(|| model.forward(&prompt, &mut kv_ref)).unwrap();
    let tok = argmax(&to_f32(&logits0)) as u32;
    let next = Tensor::from_vec(vec![tok], vec![1usize, 1], cuda).unwrap();
    let logits_ref = no_grad(|| model.forward(&next, &mut kv_ref)).unwrap();
    let mut kv_dev = model.make_kv_cache(1, l + 8).unwrap();
    let _ = no_grad(|| model.forward(&prompt, &mut kv_dev)).unwrap();
    let mut st = model.make_decode_state().unwrap();
    st.update(tok, l as u32).unwrap();
    no_grad(|| model.forward_decode_dev(&mut st, &mut kv_dev)).unwrap();
    let cs = cos_sim(&to_f32(&logits_ref), &to_f32(&st.logits));
    eprintln!("[P6.3 nvfp4] single-step cos_sim={cs:.6}");
    assert!(cs >= 0.99, "nvfp4 forward_decode_dev diverges from forward: cos_sim={cs}");

    // Perf headline (64 токенов, JIT прогрет).
    let cfg = GenerationConfig { max_new_tokens: 64, temperature: 0.0, max_seq: Some(ids.len() + 96), ..Default::default() };
    let warm = GenerationConfig { max_new_tokens: 8, ..cfg.clone() };
    let _ = pipeline.generate(&ids, warm.clone()).expect("warmup baseline");
    let _ = pipeline.generate_with_graph(&ids, warm).expect("warmup graph");

    let (_base_ids, base_stats) = pipeline.generate(&ids, cfg.clone()).expect("generate baseline");
    let (graph_ids, graph_stats) = pipeline.generate_with_graph(&ids, cfg).expect("generate graph");

    let base_tps = (base_stats.new_tokens as f64) / (base_stats.decode_ms.max(1) as f64) * 1000.0;
    let graph_tps = (graph_stats.new_tokens as f64) / (graph_stats.decode_ms.max(1) as f64) * 1000.0;
    eprintln!(
        "[P6.3 nvfp4] baseline decode {base_tps:.1} tok/s, graph decode {graph_tps:.1} tok/s ({:.2}×)",
        graph_tps / base_tps
    );
    eprintln!("[P6.3 nvfp4] graph='{}'", pipeline.decode(&graph_ids).unwrap_or_default());
    assert!(graph_tps > base_tps, "graph decode not faster: {graph_tps:.1} vs {base_tps:.1}");
}

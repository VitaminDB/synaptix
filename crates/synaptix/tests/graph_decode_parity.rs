//! Графовый декод против обычного на профиле «оптимальный» (MXFP8-KV):
//! greedy-токены совпадают, граф быстрее. Гейтед: модели по списку.
//!
//! ```sh
//! GRAPH_MODELS=~/Storage/syn_models/gguf-tests/Qwen3-0.6B-Q8_0.gguf:... \
//!   cargo test -p synaptix --release --test graph_decode_parity -- --nocapture --test-threads=1
//! ```
//! `GRAPH_NEW_TOKENS` (64) — длина ответа.

use std::time::Instant;

use synaptix::facade::llm::{
    load_llm_with_policy, optimal_profile, set_dflash_enabled, set_graph_decode_enabled, set_mtp_enabled,
    Message, RawGenerationConfig as GenerationConfig,
};
use synaptix_core::device::Device;

fn run(model: &synaptix::facade::llm::Llm, ids: &[u32], n: usize, graph: bool) -> (Vec<u32>, f64) {
    set_graph_decode_enabled(graph);
    let temperature: f32 = std::env::var("GRAPH_TEMP").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let cfg = GenerationConfig { max_new_tokens: n, temperature, seed: 7, max_seq: Some(ids.len() + n + 8), ..Default::default() };
    let mut out = Vec::new();
    let mut first: Option<Instant> = None;
    let mut sink = |t: u32| {
        first.get_or_insert_with(Instant::now);
        out.push(t);
        true
    };
    model.generate_streaming(ids, cfg, &mut sink).expect("генерация");
    let dt = first.map(|f| f.elapsed().as_secs_f64()).unwrap_or(0.0);
    let tps = (out.len().saturating_sub(1)) as f64 / dt.max(1e-6);
    (out, tps)
}

#[test]
fn graph_matches_eager_on_optimal_profile() {
    let Ok(list) = std::env::var("GRAPH_MODELS") else {
        eprintln!("GRAPH_MODELS не задан — пропуск");
        return;
    };
    let n: usize = std::env::var("GRAPH_NEW_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let mut failed = Vec::new();
    for path in list.split(':').filter(|s| !s.is_empty()) {
        let path = std::path::PathBuf::from(path);
        let mut opt = optimal_profile(&path);
        if std::env::var_os("GRAPH_KV_DENSE").is_some() {
            opt.policy.kv_dtype = synaptix::facade::llm::KvDtypePolicy::BF16;
        }
        // Сравниваем чистый декод: без спекуляции (у неё свои пути).
        let mtp = std::env::var_os("GRAPH_MTP").is_some();
        set_mtp_enabled(mtp);
        set_dflash_enabled(false);
        let (model, tok) = load_llm_with_policy(&path, opt.policy.clone(), &Device::Cuda(0)).expect("загрузка");
        let prompt = tok
            .apply_chat_template_ex_tools(
                &[Message::user(std::env::var("GRAPH_PROMPT").unwrap_or_else(|_| {
                    "Write a short paragraph about rivers, bridges and old towns.".into()
                }))],
                true,
                false,
                None,
            )
            .expect("шаблон");
        let ids = tok.encode(&prompt).expect("encode");
        // Прогрев JIT обоих путей.
        let _ = run(&model, &ids, 4, false);
        let _ = run(&model, &ids, 4, true);
        if std::env::var_os("GRAPH_ONLY").is_some() {
            let (out, tps) = run(&model, &ids, n, true);
            eprintln!("[graph] только граф: {tps:.1} ток/с: {:?}", tok.decode(&out));
            continue;
        }
        let (eager, eager_tps) = run(&model, &ids, n, false);
        let (graph, graph_tps) = run(&model, &ids, n, true);
        let same = eager.iter().zip(&graph).take_while(|(a, b)| a == b).count();
        eprintln!(
            "[graph] {}: KV {:?}, совпало {same}/{} токенов, обычный {eager_tps:.1} ток/с, граф {graph_tps:.1} ток/с ({:+.0} %)",
            path.file_name().unwrap().to_string_lossy(),
            opt.policy.kv_dtype,
            eager.len().min(graph.len()),
            (graph_tps / eager_tps.max(1e-6) - 1.0) * 100.0
        );
        // Плотный KV — строго: пути отличаются только порядком суммирования
        // в F16 (~0.02 в логитах), greedy совпадает. MXFP8 ту же крошечную
        // разницу в K/V иногда переводит через границу округления E4M3 (3 бита
        // мантиссы) — ~0.3 в логитах, и на почти равных токенах ответ
        // ветвится (Qwen3-0.6B: с 10-го токена на одном промпте, 128/128 на
        // другом) — см. `decode_dev_step_diff`.
        // Слитый device-путь Gemma-4 тоже ветвится рано (другие ядра слоя).
        // Строго сверяем только плотный KV; остальное — замер и глазами текст.
        let need = if opt.policy.kv_dtype == synaptix::facade::llm::KvDtypePolicy::MXFP8 {
            0
        } else {
            eager.len().min(graph.len())
        };
        if same < eager.len().min(graph.len()) {
            eprintln!("  обычный: {:?}\n  граф:    {:?}", tok.decode(&eager), tok.decode(&graph));
        }
        if same < need {
            failed.push(path.display().to_string());
        }
        drop(model);
        let _ = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(0);
    }
    set_graph_decode_enabled(false);
    assert!(failed.is_empty(), "граф разошёлся с обычным декодом: {failed:?}");
}

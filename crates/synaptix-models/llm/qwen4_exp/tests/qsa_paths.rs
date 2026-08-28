//! Разреженный путь QSA против полного внимания с маской выбора.
//!
//! Тест живёт отдельным бинарём: он переключает путь переменной окружения, а
//! она видна всему процессу.

use std::path::PathBuf;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_llm_qwen4_exp::{Qwen4ExpModel, Qwen4ExpWeights};

fn ref_dir() -> Option<PathBuf> {
    let p = PathBuf::from(std::env::var("SYN_QWEN4EXP_REF").ok()?);
    p.join("reference.safetensors").exists().then_some(p)
}

fn tokens_of(dir: &PathBuf) -> Vec<u32> {
    let raw = std::fs::read(dir.join("tokens.json")).expect("tokens.json");
    let v: serde_json::Value = serde_json::from_slice(&raw).unwrap();
    v["tokens"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as u32).collect()
}

fn host(t: &synaptix_core::tensor::Tensor) -> Vec<f32> {
    t.to_device(Device::Cpu)
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.flatten_all())
        .and_then(|x| x.to_vec1::<f32>())
        .expect("на хост")
}

fn diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)
}

/// Разреженное внимание против полного с маской выбора: пути должны совпасть.
/// На реальном бюджете (512 блоков) селекция не включается на коротком
/// эталоне, поэтому бюджет урезается переменной окружения — тогда работают и
/// сборка блоками, и тайлы объединения.
#[test]
fn cuda_sparse_gather_matches_selection_mask() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
    let device = Device::Cuda(0);
    let dtype = DType::F32;
    let weights = Qwen4ExpWeights::open(&dir, device, dtype).expect("open");
    let cfg = weights.config.clone();
    let model = Qwen4ExpModel::build(
        &cfg,
        &weights,
        device,
        dtype,
        dtype,
        cfg.max_position_embeddings.min(4096),
        &|layer| weights.ngram_rows(layer),
    )
    .expect("build");
    let tokens = tokens_of(&dir);

    let run = || {
        let mut cache = model.make_cache(tokens.len() + 8).expect("кэш");
        host(&model.forward(&tokens, &mut cache).expect("forward"))
    };

    std::env::set_var("SYN_QWEN4EXP_QSA_TOPK", "1");
    std::env::set_var("SYN_QWEN4EXP_QSA_GATHER", "1");
    let gathered = run();
    std::env::set_var("SYN_QWEN4EXP_QSA_GATHER", "0");
    let masked = run();
    std::env::remove_var("SYN_QWEN4EXP_QSA_GATHER");
    std::env::remove_var("SYN_QWEN4EXP_QSA_TOPK");

    // Контроль: без урезания бюджета селекция не включается вовсе, и ответ
    // обязан отличаться — иначе сравнивать было нечего.
    let full = run();

    let d = diff(&gathered, &masked);
    let scale = masked.iter().map(|v| v.abs()).fold(0f32, f32::max);
    let against_full = diff(&gathered, &full);
    eprintln!(
        "сборка против маски: max_abs={d:.3e}, против полного внимания {against_full:.3e}, масштаб {scale:.3}"
    );
    assert!(against_full > 1e-4, "селекция не включилась: пути совпали с полным вниманием");
    assert!(d < 2e-3 * scale.max(1.0), "разреженные пути разошлись: {d:.3e}");
}

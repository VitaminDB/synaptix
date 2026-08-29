//! Спекулятивный шаг обязан быть невидим для результата.
//!
//! Прогон пары «токен + драфт» рвёт скан линейного внимания и свёртку PLE
//! посередине, а при отказе возвращает состояние на границу. Проверяется, что
//! логиты первой позиции пары совпадают с обычным декодом, и что после отката
//! продолжение идёт так, будто драфта не было вовсе.

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

fn check(device: Device, dtype: DType, tol: f32) {
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
    eprintln!("эталон {}", dir.display());
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
    assert!(tokens.len() >= 6, "эталон слишком короткий");
    let head = tokens.len() - 4;

    let mut plain = model.make_cache(tokens.len() + 8).expect("кэш");
    model.forward(&tokens[..head], &mut plain).expect("префилл");
    let mut expected = Vec::new();
    for t in &tokens[head..] {
        expected.push(host(&model.forward(&[*t], &mut plain).expect("шаг")));
    }

    let mut spec = model.make_cache(tokens.len() + 8).expect("кэш");
    model.forward(&tokens[..head], &mut spec).expect("префилл");

    // Драфт заведомо неверный: интересен именно откат.
    let wrong = tokens[head].wrapping_add(1) % cfg.vocab_size as u32;
    let (hidden, _, snap) = model
        .forward_pair(
            &[tokens[head], wrong],
            &mut spec,
            synaptix_llm_common::model::RopePositions::Sequential,
        )
        .expect("пара");
    let first = host(&model.lm_head_forward(&hidden.narrow(0, 0, 1).unwrap().contiguous().unwrap()).unwrap());
    let d0 = diff(&first, &expected[0]);
    assert!(d0 < tol, "{device:?}/{dtype:?}: логиты первой позиции пары разошлись: {d0:.3e}");

    spec.restore(&snap).expect("откат");
    assert_eq!(spec.seq_len, snap.seq_len(), "откат не вернул длину");
    for (i, t) in tokens[head + 1..].iter().enumerate() {
        let got = host(&model.forward(&[*t], &mut spec).expect("шаг после отката"));
        let d = diff(&got, &expected[i + 1]);
        assert!(d < tol, "{device:?}/{dtype:?}: шаг {i} после отката разошёлся: {d:.3e}");
    }

    // Принятый драфт: вторая позиция пары должна совпасть со своим обычным шагом.
    let mut kept = model.make_cache(tokens.len() + 8).expect("кэш");
    model.forward(&tokens[..head], &mut kept).expect("префилл");
    let (hidden, _, _) = model
        .forward_pair(
            &[tokens[head], tokens[head + 1]],
            &mut kept,
            synaptix_llm_common::model::RopePositions::Sequential,
        )
        .expect("пара");
    let second = host(&model.lm_head_forward(&hidden.narrow(0, 1, 1).unwrap().contiguous().unwrap()).unwrap());
    let d1 = diff(&second, &expected[1]);
    assert!(d1 < tol, "{device:?}/{dtype:?}: вторая позиция пары разошлась: {d1:.3e}");
}

#[test]
fn cpu_pair_matches_step_by_step() {
    synaptix_kernels_cpu::ensure_registered();
    check(Device::Cpu, DType::F32, 2e-3);
}

#[test]
fn cuda_pair_matches_step_by_step() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    check(Device::Cuda(0), DType::F32, 2e-3);
}

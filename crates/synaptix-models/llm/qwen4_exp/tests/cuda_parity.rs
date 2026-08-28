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

fn build(dir: &PathBuf, device: Device, dtype: DType) -> (Qwen4ExpWeights, Qwen4ExpModel) {
    let weights = Qwen4ExpWeights::open(dir, device, dtype).expect("open");
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
    (weights, model)
}

fn run(dir: &PathBuf, device: Device, dtype: DType) -> Vec<f32> {
    let weights = Qwen4ExpWeights::open(dir, device, dtype).expect("open");
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
    let tokens = tokens_of(dir);
    let mut cache = model.make_cache(tokens.len() + 8).expect("cache");
    model
        .forward(&tokens, &mut cache)
        .expect("forward")
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap()
}

/// Декод по одному токену обязан совпасть с префиллом всей последовательности:
/// это ловит и рассинхрон состояния линейного внимания, и то, что скан на
/// декоде идёт по фактической длине чанка, а не по дополненной.
#[test]
fn cuda_decode_matches_prefill() {
    let Some(dir) = ref_dir() else {
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    let device = Device::Cuda(0);
    let tokens = tokens_of(&dir);
    let (_w, model) = build(&dir, device, DType::F32);

    let mut prefill_cache = model.make_cache(tokens.len() + 8).expect("cache");
    let prefill = model
        .forward(&tokens, &mut prefill_cache)
        .expect("prefill")
        .to_device(Device::Cpu)
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .unwrap();

    let mut cache = model.make_cache(tokens.len() + 8).expect("cache");
    let mut step = Vec::new();
    for token in &tokens {
        let out = model.forward(&[*token], &mut cache).expect("step");
        step.extend(
            out.to_device(Device::Cpu)
                .and_then(|t| t.to_dtype(DType::F32))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .unwrap(),
        );
    }

    assert_eq!(step.len(), prefill.len());
    let max_abs = step
        .iter()
        .zip(&prefill)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let scale = prefill.iter().map(|v| v.abs()).fold(0f32, f32::max);
    eprintln!("CUDA декод против префилла: max_abs={max_abs:.3e}, масштаб {scale:.3}");
    assert!(max_abs < 2e-3 * scale.max(1.0), "декод разошёлся: {max_abs:.3e}");
}

#[test]
fn cuda_matches_cpu() {
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }

    let cpu = run(&dir, Device::Cpu, DType::F32);
    for dtype in [DType::F32, DType::F16] {
        let gpu = run(&dir, Device::Cuda(0), dtype);
        assert_eq!(cpu.len(), gpu.len());
        let (mut max_abs, mut worst) = (0f32, 0usize);
        for (i, (a, b)) in gpu.iter().zip(&cpu).enumerate() {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
                worst = i;
            }
        }
        let scale = cpu.iter().map(|v| v.abs()).fold(0f32, f32::max);
        eprintln!("CUDA {dtype:?}: max_abs={max_abs:.3e} (позиция {worst}), масштаб {scale:.3}");
        let tol = if dtype == DType::F32 { 2e-3 } else { 6e-2 };
        assert!(max_abs < tol * scale.max(1.0), "CUDA {dtype:?}: max_abs={max_abs:.3e}");
    }
}

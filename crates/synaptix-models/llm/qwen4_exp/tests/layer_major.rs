//! Послойный префилл против обычного.
//!
//! Порядок обхода меняется — сперва слой, потом чанки промпта, — но состояния
//! обновляются теми же кусками в том же порядке, так что скрытое состояние
//! последней позиции обязано совпасть с точностью до порядка суммирования.

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

fn check(device: Device, dtype: DType, chunk: usize, tol: f32) {
    let Some(dir) = ref_dir() else {
        eprintln!("SYN_QWEN4EXP_REF не задан — пропуск");
        return;
    };
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

    let mut plain = model.make_cache(tokens.len() + 8).expect("кэш");
    let hidden = model.forward_hidden(&tokens, &mut plain).expect("обычный префилл");
    let last = hidden.dims()[0] - 1;
    let want = host(&hidden.narrow(0, last, 1).unwrap().contiguous().unwrap());

    let mut layered = model.make_cache(tokens.len() + 8).expect("кэш");
    let (got, _) = model
        .prefill_by_layers(&tokens, &[], &mut layered, chunk)
        .expect("послойный префилл");
    let got = host(&got);

    assert_eq!(got.len(), want.len());
    let max_abs = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let scale = want.iter().map(|v| v.abs()).fold(0f32, f32::max);
    eprintln!("послойно против обычного ({chunk}): max_abs={max_abs:.3e}, масштаб {scale:.3}");
    assert!(max_abs < tol * scale.max(1.0), "разошлось: {max_abs:.3e}");
    assert_eq!(layered.seq_len, plain.seq_len, "длина кэша разъехалась");
}

#[test]
fn cpu_layer_major_matches_plain() {
    synaptix_kernels_cpu::ensure_registered();
    check(Device::Cpu, DType::F32, 4, 2e-3);
}

#[test]
fn cuda_layer_major_matches_plain() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    if synaptix_core::device::cuda::get(0).is_err() {
        eprintln!("CUDA-устройств нет — пропуск");
        return;
    }
    check(Device::Cuda(0), DType::F32, 4, 2e-3);
}

//! Гейт CTC-дообучения против PyTorch: loss и градиенты верхних слоёв.
//!
//! Эталон: `reference/dump_ctc_grads.py` (поверх `dump_wav2vec2.py`) →
//! `tmp/w2v2_ref/grad_*.npy`, `grad_meta.json`. Модель — `W2V_MODEL`
//! (по умолчанию `models/w2v2-lv60-espeak`). Без артефактов — SKIP.

use std::path::PathBuf;

use synaptix_asr_wav2vec2::{normalize_audio, Wav2Vec2Weights, Wav2Vec2Model};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::log_softmax_dim;
use synaptix_train::losses::{ctc_loss, CtcReduction};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../..")
}

fn model_dir() -> PathBuf {
    std::env::var("W2V_MODEL").map(PathBuf::from).unwrap_or_else(|_| root().join("models/w2v2-lv60-espeak"))
}

fn ref_dir() -> PathBuf {
    std::env::var("W2V_REF").map(PathBuf::from).unwrap_or_else(|_| root().join("tmp/w2v2_ref"))
}

fn load_npy(path: &PathBuf) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let (hlen, off) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[off..off + hlen]).unwrap();
    assert!(header.contains("<f4") && !header.contains("'fortran_order': True"), "{header}");
    let i = header.find("'shape'").unwrap();
    let rest = &header[i..];
    let shape = rest[rest.find('(').unwrap() + 1..rest.find(')').unwrap()]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = bytes[off + hlen..].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    (shape, data)
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().contiguous().unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += x as f64 * y as f64;
        na += (x as f64).powi(2);
        nb += (y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn run(device: Device) {
    synaptix_autograd::init().unwrap();
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(ref_dir().join("grad_meta.json")).unwrap()).unwrap();
    let train: Vec<String> = meta["train"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let check: Vec<String> = meta["check"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    let labels: Vec<u32> = meta["labels"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();

    let w = Wav2Vec2Weights::from_unpacked(model_dir(), device, DType::F32).unwrap();
    let mut model = Wav2Vec2Model::load(&w).unwrap();
    let mut handles = Vec::new();
    for (name, t) in model.named_parameters_mut() {
        if train.iter().any(|p| name.starts_with(p.as_str())) {
            *t = t.clone().requires_grad_(true);
            handles.push((name, t.clone()));
        }
    }
    let (_, wav) = load_npy(&ref_dir().join("wav16.npy"));
    let x = normalize_audio(&wav);
    let n = x.len();
    let x = Tensor::from_vec(x, (1, n), device).unwrap();
    let logits = model.logits(&x).unwrap();
    let lp = log_softmax_dim(&logits.to_dtype(DType::F32).unwrap(), 2).unwrap();
    let frames = lp.dims()[1];
    let loss = ctc_loss(&lp, &[frames], &[labels], 0, CtcReduction::Sum, false).unwrap();
    let loss_v = host(&loss)[0] as f64;
    let ref_loss = meta["loss"].as_f64().unwrap();
    eprintln!("[ctc_grad] {device:?} loss {loss_v:.5} (ref {ref_loss:.5})");
    assert!((loss_v - ref_loss).abs() < 1e-3 * ref_loss.max(1.0), "loss {loss_v} vs {ref_loss}");
    loss.backward().unwrap();
    for name in check {
        let (_, t) = handles.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("нет параметра {name}"));
        let g = t.grad().unwrap_or_else(|| panic!("нет градиента {name}"));
        let (shape, r) = load_npy(&ref_dir().join(format!("grad_{name}.npy")));
        assert_eq!(g.dims(), &shape[..], "{name}");
        let c = cosine(&host(&g), &r);
        let norm = |v: &[f32]| v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let ratio = norm(&host(&g)) / norm(&r).max(1e-30);
        eprintln!("[ctc_grad]   {name:<60} cos {c:.6} |g|/|ref| {ratio:.5}");
        assert!(c > 0.999 && (ratio - 1.0).abs() < 0.01, "{name}: cos {c}, ratio {ratio}");
    }
}

fn have() -> bool {
    let ok = ref_dir().join("grad_meta.json").exists() && model_dir().join("model.safetensors").exists();
    if !ok {
        eprintln!("[ctc_grad] SKIP: нет эталона или модели");
    }
    ok
}

#[test]
fn ctc_grad_gate_cpu() {
    if !have() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    run(Device::Cpu);
}

#[test]
fn ctc_grad_gate_cuda() {
    if !have() || std::env::var("SYN_SKIP_CUDA").is_ok() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    run(Device::Cuda(0));
}

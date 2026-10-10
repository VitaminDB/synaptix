//! Кросс-фреймворк-гейт Wav2Vec2ForCTC против HF transformers (CPU F32).
//!
//! Эталон: `reference/dump_wav2vec2.py` → `tmp/w2v2_ref/` в корне workspace
//! (wav16, features, proj, pos, hidden, logits, text.json); модель —
//! `models/w2v2-xlsr53-espeak/`. Пути переопределяются `W2V_MODEL` / `W2V_REF`.
//! Гейт: per-row cosine ≥ 0.999 на каждой стадии и совпадение greedy-фонем.
//! Без артефактов на диске — SKIP.

use std::path::PathBuf;

use synaptix_asr_wav2vec2::Wav2Vec2Ctc;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../..")
}

fn model_dir() -> PathBuf {
    std::env::var("W2V_MODEL").map(PathBuf::from).unwrap_or_else(|_| root().join("models/w2v2-xlsr53-espeak"))
}

fn ref_dir() -> PathBuf {
    std::env::var("W2V_REF").map(PathBuf::from).unwrap_or_else(|_| root().join("tmp/w2v2_ref"))
}

/// Минимальный разбор C-order f32 `.npy`.
fn load_npy(path: &PathBuf) -> (Vec<usize>, Vec<f32>) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(&bytes[0..6], b"\x93NUMPY");
    let (hlen, off) = if bytes[6] == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize, 12)
    };
    let header = std::str::from_utf8(&bytes[off..off + hlen]).unwrap();
    assert!(header.contains("<f4"), "{header}");
    assert!(!header.contains("'fortran_order': True"), "{header}");
    let i = header.find("'shape'").unwrap();
    let rest = &header[i..];
    let shape: Vec<usize> = rest[rest.find('(').unwrap() + 1..rest.find(')').unwrap()]
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = bytes[off + hlen..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    (shape, data)
}

fn host(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32).unwrap().to_device(Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

/// Минимальный per-row cosine по последней оси.
fn min_row_cos(got: &[f32], reference: &[f32], inner: usize) -> (f32, usize) {
    assert_eq!(got.len(), reference.len());
    let mut worst = (f32::INFINITY, 0);
    for (r, (a, b)) in got.chunks(inner).zip(reference.chunks(inner)).enumerate() {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            dot += x as f64 * y as f64;
            na += (x as f64).powi(2);
            nb += (y as f64).powi(2);
        }
        let c = (dot / (na.sqrt() * nb.sqrt()).max(1e-12)) as f32;
        if c < worst.0 {
            worst = (c, r);
        }
    }
    worst
}

fn check(tag: &str, got: &Tensor, file: &str, min_cos: f32) {
    let (shape, reference) = load_npy(&ref_dir().join(file));
    assert_eq!(got.dims(), &shape[..], "{tag}: shape");
    let inner = *shape.last().unwrap();
    let (c, row) = min_row_cos(&host(got), &reference, inner);
    eprintln!("[w2v2_gate] {tag:<8} min_cos={c:.6} (row {row})");
    assert!(c >= min_cos, "{tag}: cosine {c} < {min_cos} (row {row})");
}

fn run_gate(device: Device, dtype: DType, min_cos: f32) {
    let model = Wav2Vec2Ctc::from_unpacked(model_dir(), &device, dtype).expect("load");
    let (_, wav) = load_npy(&ref_dir().join("wav16.npy"));
    let tr = model.forward_trace(&wav).expect("forward");
    check("features", &tr.features, "features.npy", min_cos);
    check("proj", &tr.proj, "proj.npy", min_cos);
    check("pos", &tr.pos, "pos.npy", min_cos);
    check("hidden", &tr.hidden, "hidden.npy", min_cos);
    check("logits", &tr.logits, "logits.npy", min_cos);

    let got = model.recognize(&wav, 16000).expect("recognize");
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(ref_dir().join("text.json")).unwrap()).unwrap();
    let reference: Vec<String> = reference["phonemes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    eprintln!("[w2v2_gate] got: {}", got.join(" "));
    eprintln!("[w2v2_gate] ref: {}", reference.join(" "));
    assert_eq!(got, reference, "greedy-фонемы расходятся");
}

fn have_artifacts() -> bool {
    let ok = model_dir().join("model.safetensors").exists() && ref_dir().join("logits.npy").exists();
    if !ok {
        eprintln!("[w2v2_gate] SKIP: нет модели или эталона");
    }
    ok
}

#[test]
fn wav2vec2_gate_cpu_f32() {
    if !have_artifacts() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    run_gate(Device::Cpu, DType::F32, 0.999);
}

#[test]
fn wav2vec2_gate_cuda_f32() {
    if !have_artifacts() || std::env::var("SYN_SKIP_CUDA").is_ok() {
        return;
    }
    synaptix_kernels_cuda::ensure_registered();
    run_gate(Device::Cuda(0), DType::F32, 0.999);
}

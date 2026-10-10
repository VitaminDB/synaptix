//! Гейт WavLM-Large (базовая модель, без CTC-головы) против transformers:
//! features / proj / pos / hidden. Эталон — `reference/dump_wavlm.py` →
//! `tmp/wavlm_ref/`, модель — `models/wavlm-large`. Без артефактов — SKIP.

use std::path::PathBuf;

use synaptix_asr_wav2vec2::{normalize_audio, Wav2Vec2Model, Wav2Vec2Weights};
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../..")
}

fn ref_dir() -> PathBuf {
    root().join("tmp/wavlm_ref")
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
    eprintln!("[wavlm_gate] {tag:<8} min_cos={c:.6} (row {row})");
    assert!(c >= min_cos, "{tag}: cosine {c} < {min_cos} (row {row})");
}


fn run(device: Device) {
    let dir = root().join("models/wavlm-large");
    if !dir.join("model.safetensors").exists() || !ref_dir().join("hidden.npy").exists() {
        eprintln!("[wavlm_gate] SKIP");
        return;
    }
    let w = Wav2Vec2Weights::from_unpacked(&dir, device, DType::F32).unwrap();
    let m = Wav2Vec2Model::load(&w).unwrap();
    let (_, wav) = load_npy(&ref_dir().join("wav16.npy"));
    let x = normalize_audio(&wav);
    let n = x.len();
    let x = Tensor::from_vec(x, (1, n), device).unwrap();
    let f = m.extract_features(&x).unwrap();
    let p = m.project(&f).unwrap();
    let (pos, h) = m.encode_with_pos(&p).unwrap();
    check("features", &f, "features.npy", 0.999);
    check("proj", &p, "proj.npy", 0.999);
    check("pos", &pos, "pos.npy", 0.999);
    check("hidden", &h, "hidden.npy", 0.999);
}

#[test]
fn wavlm_gate_cpu() {
    synaptix_kernels_cpu::ensure_registered();
    run(Device::Cpu);
}

#[test]
fn wavlm_gate_cuda() {
    if std::env::var("SYN_SKIP_CUDA").is_ok() {
        return;
    }
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    run(Device::Cuda(0));
}

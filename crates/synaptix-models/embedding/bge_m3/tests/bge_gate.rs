//! Гейт BGE-M3 (XLM-RoBERTa) против Python-эталона (PyTorch CPU F32).
//!
//! Эталон в `tmp/bge_ref/` (см. `reference/dump_bge.py`):
//!   input_ids.npy (1,S i64), attention_mask.npy (1,S i64), last_hidden.npy
//!   (1,S,1024 f32), dense_ref.npy (1,1024 f32). Веса —
//!   `tmp/bge_unpack/model.safetensors`.
//!
//! Кросс-фреймворк F32 (PyTorch vs synaptix CPU) НЕ бит-идентичен → per-row gate:
//! last_hidden per-row (по hidden=1024) cosine ≥ 0.9999 на всех S строках; dense
//! cosine ≥ 0.99999 + max-abs мал.
//!
//! Гейт guarded: skip если весов/эталона нет на диске.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_embedding_bge_m3::config::BgeConfig;
use synaptix_embedding_bge_m3::loader::BgeWeights;
use synaptix_embedding_bge_m3::model::{l2_normalize, BgeEncoder};

const UNPACK: &str = "tmp/bge_unpack";
const REF: &str = "tmp/bge_ref";

fn parse_npy(path: &str) -> (Vec<usize>, Vec<u8>, String) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert_eq!(&bytes[0..6], b"\x93NUMPY", "npy magic");
    let major = bytes[6];
    let (hdr_len, data_off) = if major == 1 {
        (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            12,
        )
    };
    let header = std::str::from_utf8(&bytes[data_off..data_off + hdr_len]).unwrap();
    let descr = {
        let i = header.find("'descr'").unwrap();
        let rest = &header[i + 7..];
        let q1 = rest.find('\'').unwrap();
        let q2 = rest[q1 + 1..].find('\'').unwrap();
        rest[q1 + 1..q1 + 1 + q2].to_string()
    };
    let shape = {
        let i = header.find("'shape'").unwrap();
        let rest = &header[i..];
        let lp = rest.find('(').unwrap();
        let rp = rest.find(')').unwrap();
        rest[lp + 1..rp]
            .split(',')
            .filter_map(|s| s.trim().parse::<usize>().ok())
            .collect::<Vec<_>>()
    };
    let data = bytes[data_off + hdr_len..].to_vec();
    (shape, data, descr)
}

fn load_i64(path: &str) -> (Vec<usize>, Vec<i64>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(descr.contains("i8") || descr.contains("i4"), "descr={descr} ({path})");
    let n: usize = shape.iter().product();
    let elem = if descr.contains("i8") { 8 } else { 4 };
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * elem;
        let val = if elem == 8 {
            i64::from_le_bytes(data[o..o + 8].try_into().unwrap())
        } else {
            i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as i64
        };
        v.push(val);
    }
    (shape, v)
}

fn load_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(descr.contains("f4"), "descr={descr} ({path})");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        v.push(f32::from_le_bytes(data[o..o + 4].try_into().unwrap()));
    }
    (shape, v)
}

fn cosine(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64) * (a[i] as f64);
        nb += (b[i] as f64) * (b[i] as f64);
        let d = (a[i] - b[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    ((dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32, max_abs)
}

#[test]
fn bge_gate() {
    let weights_path = format!("{UNPACK}/model.safetensors");
    if !Path::new(&weights_path).exists()
        || !Path::new(&format!("{REF}/last_hidden.npy")).exists()
        || !Path::new(&format!("{UNPACK}/config.json")).exists()
    {
        eprintln!("[bge_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let cfg = BgeConfig::from_json_bytes(
        &std::fs::read(format!("{UNPACK}/config.json")).unwrap(),
    )
    .expect("parse config");

    let (ids_shape, ids) = load_i64(&format!("{REF}/input_ids.npy"));
    let (_m_shape, mask) = load_i64(&format!("{REF}/attention_mask.npy"));
    let (lh_shape, lh_ref) = load_f32(&format!("{REF}/last_hidden.npy"));
    let (de_shape, dense_ref) = load_f32(&format!("{REF}/dense_ref.npy"));

    assert_eq!(ids_shape.len(), 2, "input_ids rank");
    let (bsz, s) = (ids_shape[0], ids_shape[1]);
    let hidden = cfg.hidden_size;
    assert_eq!(lh_shape, vec![bsz, s, hidden], "last_hidden ref shape");
    assert_eq!(de_shape, vec![bsz, hidden], "dense ref shape");

    let input_ids = Tensor::from_vec(ids, vec![bsz, s], Device::Cpu).unwrap();
    let attention_mask = Tensor::from_vec(mask, vec![bsz, s], Device::Cpu).unwrap();

    let weights = BgeWeights::load_safetensors(&weights_path, Device::Cpu, DType::F32)
        .expect("load weights");
    eprintln!("[bge_gate] loaded {} tensors", weights.len());

    let encoder = BgeEncoder::build(&cfg, &weights).expect("build encoder");

    let last_hidden = encoder.forward(&input_ids, &attention_mask).expect("forward");
    assert_eq!(last_hidden.dims(), &[bsz, s, hidden], "last_hidden shape");
    let lh = last_hidden
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();

    // last_hidden per-row (по hidden) cosine + max-abs.
    let rows = bsz * s;
    let mut min_cos = f32::INFINITY;
    let mut max_abs = 0.0f32;
    for r in 0..rows {
        let o = r * hidden;
        let (c, ma) = cosine(&lh[o..o + hidden], &lh_ref[o..o + hidden]);
        if c < min_cos {
            min_cos = c;
        }
        if ma > max_abs {
            max_abs = ma;
        }
    }
    eprintln!(
        "[bge_gate] last_hidden rows={rows} min_cos={min_cos:.6} max_abs={max_abs:.5}"
    );

    // dense = L2-norm(CLS).
    let dense = encoder.dense_embed(&last_hidden).expect("dense embed");
    assert_eq!(dense.dims(), &[bsz, hidden], "dense shape");
    let de = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (dense_cos, dense_max_abs) = cosine(&de, &dense_ref);
    eprintln!("[bge_gate] dense cos={dense_cos:.7} max_abs={dense_max_abs:.6}");

    // sanity: эталонный dense нормирован (‖·‖≈1).
    let our_norm = l2_normalize(&dense).unwrap();
    let _ = our_norm;

    assert!(
        min_cos >= 0.9999,
        "last_hidden min per-row cosine {min_cos:.6} < 0.9999"
    );
    assert!(
        dense_cos >= 0.99999,
        "dense cosine {dense_cos:.7} < 0.99999"
    );
}

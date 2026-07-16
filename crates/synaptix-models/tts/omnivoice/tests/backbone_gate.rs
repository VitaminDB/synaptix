//! Bit-exact-гейт backbone OmniVoice против Python-эталона (PyTorch CPU F32).
//!
//! Эталон в `tmp/ov_ref/` (см. `reference/dump_backbone.py`):
//!   input_ids.npy (1,8,S i64), audio_mask.npy (1,S bool), audio_logits.npy
//!   (1,8,S,1025 f32). Веса — `tmp/ov_unpack/model.safetensors`.
//!
//! Кросс-фреймворк F32 (PyTorch vs synaptix CPU) НЕ бит-идентичен → гейт per-row
//! (по последней оси V=1025): cosine ≥ 0.9999 на ВСЕХ (8·S) строках + per-row
//! max-abs мал относительно масштаба логитов (mean ~59).
//!
//! Гейт guarded: skip (return) если весов/эталона нет на диске.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_omnivoice::backbone::Backbone;
use synaptix_tts_omnivoice::config::OmniVoiceConfig;
use synaptix_tts_omnivoice::loader::OmniVoiceLmWeights;

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";

/// Минимальный npy-парсер: magic(6) + ver(2) + hdr_len(2 LE) + dict-header + raw.
/// Возвращает (shape, raw_data_bytes, descr_string).
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
    // descr
    let descr = {
        let i = header.find("'descr'").unwrap();
        let rest = &header[i + 7..];
        let q1 = rest.find('\'').unwrap();
        let q2 = rest[q1 + 1..].find('\'').unwrap();
        rest[q1 + 1..q1 + 1 + q2].to_string()
    };
    // shape tuple
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

fn load_input_ids() -> Tensor {
    let (shape, data, descr) = parse_npy(&format!("{REF}/input_ids.npy"));
    assert!(descr.contains("i8"), "input_ids descr={descr}");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 8;
        v.push(i64::from_le_bytes(data[o..o + 8].try_into().unwrap()));
    }
    Tensor::from_vec(v, shape, Device::Cpu).unwrap()
}

fn load_audio_mask() -> Tensor {
    let (shape, data, descr) = parse_npy(&format!("{REF}/audio_mask.npy"));
    assert!(descr.contains('b') || descr.contains("?"), "mask descr={descr}");
    let n: usize = shape.iter().product();
    let v: Vec<u8> = (0..n).map(|i| if data[i] != 0 { 1u8 } else { 0u8 }).collect();
    Tensor::from_vec(v, shape, Device::Cpu).unwrap()
}

fn load_ref_logits() -> (Vec<usize>, Vec<f32>) {
    let (shape, data, descr) = parse_npy(&format!("{REF}/audio_logits.npy"));
    assert!(descr.contains("f4"), "logits descr={descr}");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        v.push(f32::from_le_bytes(data[o..o + 4].try_into().unwrap()));
    }
    (shape, v)
}

#[test]
fn backbone_gate() {
    let weights_path = format!("{UNPACK}/model.safetensors");
    if !Path::new(&weights_path).exists()
        || !Path::new(&format!("{REF}/audio_logits.npy")).exists()
        || !Path::new(&format!("{UNPACK}/config.json")).exists()
    {
        eprintln!("[backbone_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let cfg = OmniVoiceConfig::from_json_bytes(
        &std::fs::read(format!("{UNPACK}/config.json")).unwrap(),
    )
    .expect("parse config");

    let input_ids = load_input_ids();
    let audio_mask = load_audio_mask();
    let (ref_shape, ref_flat) = load_ref_logits();

    let s = input_ids.dims()[2];
    let n_cb = cfg.num_audio_codebook;
    let av = cfg.audio_vocab_size;
    assert_eq!(ref_shape, vec![1, n_cb, s, av], "ref logits shape");

    let weights = OmniVoiceLmWeights::load_safetensors(&weights_path, Device::Cpu, DType::F32)
        .expect("load lm weights");
    eprintln!("[backbone_gate] loaded {} lm tensors", weights.len());

    let backbone = Backbone::build(&cfg, &weights, s.max(64)).expect("build backbone");

    let logits = backbone.forward(&input_ids, &audio_mask).expect("forward");
    assert_eq!(logits.dims(), &[1, n_cb, s, av], "out logits shape");
    let got = logits.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    assert_eq!(got.len(), ref_flat.len());

    // Per-row (по последней оси V=av): cosine + max-abs. Строк = n_cb*s.
    let rows = n_cb * s;
    let mut min_cos = f32::INFINITY;
    let mut max_abs = 0f32;
    let mut min_cos_row = 0usize;
    let mut max_abs_row = 0usize;
    let mut sum_abs_global = 0.0f64;
    let mut ref_abs_mean = 0.0f64;
    for r in 0..rows {
        let o = r * av;
        let a = &got[o..o + av];
        let b = &ref_flat[o..o + av];
        let mut dot = 0.0f64;
        let mut na = 0.0f64;
        let mut nb = 0.0f64;
        let mut row_max_abs = 0.0f32;
        for i in 0..av {
            dot += a[i] as f64 * b[i] as f64;
            na += (a[i] as f64) * (a[i] as f64);
            nb += (b[i] as f64) * (b[i] as f64);
            let d = (a[i] - b[i]).abs();
            if d > row_max_abs {
                row_max_abs = d;
            }
            sum_abs_global += d as f64;
            ref_abs_mean += (b[i] as f64).abs();
        }
        let cos = (dot / (na.sqrt() * nb.sqrt() + 1e-30)) as f32;
        if cos < min_cos {
            min_cos = cos;
            min_cos_row = r;
        }
        if row_max_abs > max_abs {
            max_abs = row_max_abs;
            max_abs_row = r;
        }
    }
    let mae = sum_abs_global / (rows * av) as f64;
    ref_abs_mean /= (rows * av) as f64;

    eprintln!(
        "[backbone_gate] rows={rows} min_cos={min_cos:.6} (row {min_cos_row}) \
         max_per_row_max_abs={max_abs:.4} (row {max_abs_row}) MAE={mae:.5} ref|mean|={ref_abs_mean:.3}"
    );

    assert!(
        min_cos >= 0.9999,
        "min per-row cosine {min_cos:.6} < 0.9999 (row {min_cos_row})"
    );
    assert!(
        max_abs < 0.5,
        "max per-row max-abs {max_abs:.4} >= 0.5 (row {max_abs_row})"
    );
}

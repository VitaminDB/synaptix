//! Кросс-фреймворк-гейт GigaAM-v3-e2e-CTC против PyTorch-эталона (CPU F32).
//!
//! Эталон в `tmp/gigaam_ref/` (см. `reference/dump_gigaam.py`):
//!   wav16.npy ([T] 16k f32), mel.npy (1,64,1442), encoder.npy (1,768,361),
//!   logits.npy (1,361,257), text.json. Веса/токенайзер/конфиг —
//!   `tmp/gigaam_unpack/`.
//!
//! Кросс-фреймворк F32 НЕ бит-идентичен → гейт per-row cosine (по последней оси)
//! ≥ 0.999 + точный/CER-текст. Guarded: skip если артефактов нет на диске.

use std::path::Path;

use synaptix_asr_gigaam::GigaAm;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;

const UNPACK: &str = "tmp/gigaam_unpack";
const REF: &str = "tmp/gigaam_ref";

fn parse_npy(path: &str) -> (Vec<usize>, Vec<u8>, String, bool) {
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
    let fortran = header.find("'fortran_order'").map_or(false, |i| {
        header[i..].find("True").is_some_and(|t| {
            // ближайшее True/False после ключа.
            let f = header[i..].find("False");
            f.map_or(true, |fp| t < fp)
        })
    });
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
    (shape, data, descr, fortran)
}

/// Загрузка f32-npy в C-order flat (учитывает `fortran_order`: column-major →
/// row-major перестановкой индексов).
fn load_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data, descr, fortran) = parse_npy(path);
    assert!(descr.contains("f4"), "{path} descr={descr}");
    let n: usize = shape.iter().product();
    let raw: Vec<f32> = (0..n)
        .map(|i| f32::from_le_bytes(data[i * 4..i * 4 + 4].try_into().unwrap()))
        .collect();
    if !fortran || shape.len() < 2 {
        return (shape, raw);
    }
    // Fortran (column-major) → C (row-major): out[c_index] = raw[f_index].
    let ndim = shape.len();
    let mut c_strides = vec![1usize; ndim];
    for d in (0..ndim - 1).rev() {
        c_strides[d] = c_strides[d + 1] * shape[d + 1];
    }
    let mut f_strides = vec![1usize; ndim];
    for d in 1..ndim {
        f_strides[d] = f_strides[d - 1] * shape[d - 1];
    }
    let mut out = vec![0f32; n];
    let mut idx = vec![0usize; ndim];
    for _ in 0..n {
        let mut c_off = 0;
        let mut f_off = 0;
        for d in 0..ndim {
            c_off += idx[d] * c_strides[d];
            f_off += idx[d] * f_strides[d];
        }
        out[c_off] = raw[f_off];
        // инкремент multi-index (последняя ось быстрее всего).
        for d in (0..ndim).rev() {
            idx[d] += 1;
            if idx[d] < shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    (shape, out)
}

/// Per-row cosine по последней оси `inner`. Возвращает (min_cos, argmin_row).
fn per_row_cos(got: &[f32], reference: &[f32], inner: usize) -> (f32, usize) {
    assert_eq!(got.len(), reference.len());
    let rows = got.len() / inner;
    let mut min_cos = f32::INFINITY;
    let mut min_row = 0usize;
    for r in 0..rows {
        let a = &got[r * inner..(r + 1) * inner];
        let b = &reference[r * inner..(r + 1) * inner];
        let mut dot = 0f64;
        let mut na = 0f64;
        let mut nb = 0f64;
        for k in 0..inner {
            dot += a[k] as f64 * b[k] as f64;
            na += (a[k] as f64).powi(2);
            nb += (b[k] as f64).powi(2);
        }
        let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
        let cos = (dot / denom) as f32;
        if cos < min_cos {
            min_cos = cos;
            min_row = r;
        }
    }
    (min_cos, min_row)
}

/// Levenshtein-расстояние (для CER).
fn edit_distance(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for (i, &ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, &cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

#[test]
fn gigaam_gate() {
    let weights_path = format!("{UNPACK}/model.safetensors");
    if !Path::new(&weights_path).exists() || !Path::new(&format!("{REF}/logits.npy")).exists() {
        eprintln!("[gigaam_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let model = GigaAm::from_unpacked(UNPACK, &Device::Cpu, DType::F32).expect("load model");

    let (wav_shape, wav) = load_f32(&format!("{REF}/wav16.npy"));
    eprintln!("[gigaam_gate] wav {:?}", wav_shape);

    let (mel, encoded, logits) = model.forward_debug(&wav).expect("forward_debug");

    // 1) mel.
    let (mel_shape, mel_ref) = load_f32(&format!("{REF}/mel.npy"));
    let mel_got = mel
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(mel.dims(), &mel_shape[..], "mel shape");
    let n_mels = mel_shape[1];
    let n_frames = mel_shape[2];
    // строки = n_mels (cosine по оси времени, как у Whisper-гейтов фронтенда).
    let mut mel_t = vec![0f32; mel_got.len()];
    let mut mel_ref_t = vec![0f32; mel_ref.len()];
    for m in 0..n_mels {
        for t in 0..n_frames {
            mel_t[m * n_frames + t] = mel_got[m * n_frames + t];
            mel_ref_t[m * n_frames + t] = mel_ref[m * n_frames + t];
        }
    }
    let (mel_cos, mel_row) = per_row_cos(&mel_t, &mel_ref_t, n_frames);
    eprintln!("[gigaam_gate] mel    min_cos={mel_cos:.6} (row {mel_row})");

    // 2) encoder ([1,768,361]) — cosine по оси времени (inner=T').
    let (enc_shape, enc_ref) = load_f32(&format!("{REF}/encoder.npy"));
    assert_eq!(encoded.dims(), &enc_shape[..], "encoder shape");
    let enc_got = encoded
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let enc_inner = enc_shape[2];
    let (enc_cos, enc_row) = per_row_cos(&enc_got, &enc_ref, enc_inner);
    eprintln!("[gigaam_gate] enc    min_cos={enc_cos:.6} (row {enc_row})");

    // 3) logits ([1,361,257]) — cosine по оси классов (inner=C).
    let (log_shape, log_ref) = load_f32(&format!("{REF}/logits.npy"));
    assert_eq!(logits.dims(), &log_shape[..], "logits shape");
    let log_got = logits
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let log_inner = log_shape[2];
    let (log_cos, log_row) = per_row_cos(&log_got, &log_ref, log_inner);
    eprintln!("[gigaam_gate] logits min_cos={log_cos:.6} (row {log_row})");

    // 4) text (greedy-CTC + SentencePiece) против эталона.
    let text = model.greedy_ctc_decode(&logits).expect("decode");
    let ref_text: serde_json::Value =
        serde_json::from_slice(&std::fs::read(format!("{REF}/text.json")).unwrap()).unwrap();
    let ref_text = ref_text["text"].as_str().unwrap().to_string();
    let exact = text == ref_text;
    let ref_chars: Vec<char> = ref_text.chars().collect();
    let got_chars: Vec<char> = text.chars().collect();
    let dist = edit_distance(&got_chars, &ref_chars);
    let cer = dist as f32 / ref_chars.len().max(1) as f32;
    eprintln!("[gigaam_gate] text exact={exact} CER={:.4}", cer);
    eprintln!("[gigaam_gate]   got: {text}");
    eprintln!("[gigaam_gate]   ref: {ref_text}");

    assert!(mel_cos >= 0.999, "mel cosine {mel_cos} < 0.999 (row {mel_row})");
    assert!(enc_cos >= 0.999, "encoder cosine {enc_cos} < 0.999 (row {enc_row})");
    assert!(log_cos >= 0.999, "logits cosine {log_cos} < 0.999 (row {log_row})");
    assert!(exact || cer < 0.02, "text mismatch (CER {cer})");
}

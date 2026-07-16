//! Гейт text-фронтенда OmniVoice против Python-эталона (точное совпадение id).
//!
//! Эталон в `tmp/ov_ref/`:
//!   gen_input_ids.npy (1,8,23 i64), gen_audio_mask.npy (1,23 bool) — выход
//!   `_prepare_inference_inputs("Hello world.", 12, denoise=True, остальное None)`.
//! Токенайзер — `tmp/ov_unpack/tokenizer.json`.
//!
//! Frontend строит ТЕ ЖЕ целые id (special-токены + combine + repeat по 8) →
//! гейт = ТОЧНОЕ совпадение input_ids и audio_mask.
//!
//! Гейт guarded: skip (return) если токенайзера/эталона нет на диске.

use std::path::Path;

use synaptix_tts_omnivoice::text::TextFrontend;

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";

/// Минимальный npy-парсер: magic(6)+ver(2)+hdr_len+dict-header+raw.
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
    assert!(descr.contains("i8"), "{path} descr={descr} (expect <i8)");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 8;
        v.push(i64::from_le_bytes(data[o..o + 8].try_into().unwrap()));
    }
    (shape, v)
}

fn load_bool_u8(path: &str) -> (Vec<usize>, Vec<u8>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(
        descr.contains('b') || descr.contains('?'),
        "{path} descr={descr} (expect bool)"
    );
    let n: usize = shape.iter().product();
    let v: Vec<u8> = (0..n).map(|i| if data[i] != 0 { 1u8 } else { 0u8 }).collect();
    (shape, v)
}

#[test]
fn text_gate() {
    let tok_path = format!("{UNPACK}/tokenizer.json");
    if !Path::new(&tok_path).exists()
        || !Path::new(&format!("{REF}/gen_input_ids.npy")).exists()
        || !Path::new(&format!("{REF}/gen_audio_mask.npy")).exists()
    {
        eprintln!("[text_gate] SKIP: tokenizer/ref not on disk");
        return;
    }

    // num_audio_codebook=8, audio_mask_id=1024 (см. config).
    let fe = TextFrontend::from_tokenizer_file(&tok_path, 8, 1024).expect("load tokenizer");

    let prepared = fe
        .prepare_inference_inputs("Hello world.", 12, None, None, None, None, true)
        .expect("prepare_inference_inputs");

    let got_ids = prepared
        .input_ids
        .flatten_all()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    let got_mask = prepared
        .audio_mask
        .flatten_all()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();

    let (ids_shape, ref_ids) = load_i64(&format!("{REF}/gen_input_ids.npy"));
    let (mask_shape, ref_mask) = load_bool_u8(&format!("{REF}/gen_audio_mask.npy"));

    eprintln!(
        "[text_gate] got ids dims={:?} mask dims={:?} ; ref ids {:?} mask {:?}",
        prepared.input_ids.dims(),
        prepared.audio_mask.dims(),
        ids_shape,
        mask_shape,
    );
    eprintln!("[text_gate] got row0 ids: {:?}", &got_ids[..ids_shape[2].min(got_ids.len())]);
    eprintln!("[text_gate] ref row0 ids: {:?}", &ref_ids[..ids_shape[2]]);

    assert_eq!(
        prepared.input_ids.dims(),
        &[ids_shape[0], ids_shape[1], ids_shape[2]],
        "input_ids shape mismatch"
    );
    assert_eq!(prepared.audio_mask.dims(), &[mask_shape[0], mask_shape[1]], "audio_mask shape");

    assert_eq!(got_ids.len(), ref_ids.len(), "ids length");
    let mut first_mismatch = None;
    for (i, (&g, &r)) in got_ids.iter().zip(ref_ids.iter()).enumerate() {
        if g != r {
            first_mismatch = Some((i, g, r));
            break;
        }
    }
    assert!(
        first_mismatch.is_none(),
        "input_ids mismatch at flat idx {:?} (got vs ref)",
        first_mismatch
    );

    assert_eq!(got_mask.len(), ref_mask.len(), "mask length");
    assert_eq!(got_mask, ref_mask, "audio_mask mismatch");

    eprintln!("[text_gate] EXACT MATCH input_ids ({} ids) + audio_mask ({} els)", got_ids.len(), got_mask.len());

    // Не-блокирующий лог duration-оценки для "Hello world." (auto-fallback).
    let est = synaptix_tts_omnivoice::DurationEstimator::new()
        .estimate_target_tokens("Hello world.", None, None, 1.0);
    eprintln!("[text_gate] duration estimate target_tokens(\"Hello world.\") = {est}");
}

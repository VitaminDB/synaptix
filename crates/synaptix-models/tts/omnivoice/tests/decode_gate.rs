//! Гейт masked-decode цикла OmniVoice против детерминированного Python-эталона.
//!
//! Эталон в `tmp/ov_ref/` (см. `reference/dump_generate.py`):
//!   gen_input_ids.npy (1,8,C i64), gen_audio_mask.npy (1,C bool),
//!   gen_codes.npy (8,T i64), gen_meta.json (target_len/num_step/guidance/...).
//! Веса — `tmp/ov_unpack/model.safetensors`.
//!
//! Decode greedy (position_temperature=0, class_temperature=0) → детерминизм.
//! Кросс-фреймворк argmax/topk могут разойтись на единичных неоднозначных
//! позициях из-за f32-дрейфа логитов (~1e-4); layer-penalty (±5/слой) доминирует
//! → ожидается высокое совпадение. Печатает % совпадения кодов + несовпадения.
//!
//! Гейт guarded: skip если весов/эталона нет на диске.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_omnivoice::backbone::Backbone;
use synaptix_tts_omnivoice::config::{OmniVoiceConfig, OmniVoiceGenerationConfig};
use synaptix_tts_omnivoice::loader::OmniVoiceLmWeights;
use synaptix_tts_omnivoice::masked_decode::generate_iterative;

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";

/// Минимальный npy-парсер: magic(6) + ver(2) + hdr_len + dict-header + raw.
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

fn meta_usize(meta: &str, key: &str) -> usize {
    let i = meta.find(&format!("\"{key}\"")).unwrap();
    let rest = &meta[i..];
    let colon = rest.find(':').unwrap();
    let after = &rest[colon + 1..];
    let end = after
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(after.len());
    after[..end].trim().parse::<usize>().unwrap()
}

fn meta_f32(meta: &str, key: &str) -> f32 {
    let i = meta.find(&format!("\"{key}\"")).unwrap();
    let rest = &meta[i..];
    let colon = rest.find(':').unwrap();
    let after = &rest[colon + 1..];
    let end = after
        .find(|c: char| c == ',' || c == '}' || c == '\n')
        .unwrap_or(after.len());
    after[..end].trim().parse::<f32>().unwrap()
}

#[test]
fn decode_gate() {
    let weights_path = format!("{UNPACK}/model.safetensors");
    if !Path::new(&weights_path).exists()
        || !Path::new(&format!("{REF}/gen_codes.npy")).exists()
        || !Path::new(&format!("{UNPACK}/config.json")).exists()
    {
        eprintln!("[decode_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let cfg = OmniVoiceConfig::from_json_bytes(
        &std::fs::read(format!("{UNPACK}/config.json")).unwrap(),
    )
    .expect("parse config");

    let meta = std::fs::read_to_string(format!("{REF}/gen_meta.json")).unwrap();
    let target_len = meta_usize(&meta, "target_len");
    let num_step = meta_usize(&meta, "num_step");
    let guidance_scale = meta_f32(&meta, "guidance_scale");
    let t_shift = meta_f32(&meta, "t_shift");
    let layer_penalty_factor = meta_f32(&meta, "layer_penalty_factor");

    let (ids_shape, ids_v) = load_i64(&format!("{REF}/gen_input_ids.npy"));
    let (mask_shape, mask_v) = load_bool_u8(&format!("{REF}/gen_audio_mask.npy"));
    let (codes_shape, ref_codes) = load_i64(&format!("{REF}/gen_codes.npy"));

    let n_cb = cfg.num_audio_codebook;
    let c_len = ids_shape[2];
    assert_eq!(ids_shape, vec![1, n_cb, c_len], "gen_input_ids shape");
    assert_eq!(mask_shape, vec![1, c_len], "gen_audio_mask shape");
    assert_eq!(codes_shape, vec![n_cb, target_len], "gen_codes shape");

    let cond_input_ids = Tensor::from_vec(ids_v, vec![1, n_cb, c_len], Device::Cpu).unwrap();
    let cond_audio_mask = Tensor::from_vec(mask_v, vec![1, c_len], Device::Cpu).unwrap();

    let weights = OmniVoiceLmWeights::load_safetensors(&weights_path, Device::Cpu, DType::F32)
        .expect("load lm weights");
    eprintln!("[decode_gate] loaded {} lm tensors", weights.len());

    let backbone = Backbone::build(&cfg, &weights, c_len.max(64)).expect("build backbone");

    let gen = OmniVoiceGenerationConfig {
        num_step,
        guidance_scale,
        t_shift,
        layer_penalty_factor,
        position_temperature: 0.0,
        class_temperature: 0.0,
        ..OmniVoiceGenerationConfig::default()
    };

    eprintln!(
        "[decode_gate] C={c_len} T={target_len} num_step={num_step} g={guidance_scale} \
         t_shift={t_shift} lpf={layer_penalty_factor}"
    );

    let codes = generate_iterative(
        &backbone,
        &cond_input_ids,
        &cond_audio_mask,
        target_len,
        &gen,
    )
    .expect("generate_iterative");
    assert_eq!(codes.dims(), &[n_cb, target_len], "out codes shape");
    let got = codes.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    assert_eq!(got.len(), ref_codes.len());

    // % совпадения + список несовпадений (row-major (c, t)).
    let total = got.len();
    let mut matches = 0usize;
    let mut mismatches: Vec<(usize, usize, i64, i64)> = Vec::new();
    for c in 0..n_cb {
        for j in 0..target_len {
            let idx = c * target_len + j;
            if got[idx] == ref_codes[idx] {
                matches += 1;
            } else {
                mismatches.push((c, j, got[idx], ref_codes[idx]));
            }
        }
    }
    let pct = 100.0 * matches as f64 / total as f64;
    eprintln!(
        "[decode_gate] match {matches}/{total} = {pct:.2}% ; mismatches={}",
        mismatches.len()
    );
    for (c, j, g, r) in &mismatches {
        eprintln!("  mismatch cb={c} t={j}: got={g} ref={r}");
    }

    assert!(
        pct >= 95.0,
        "code match {pct:.2}% < 95% ({} mismatches)",
        mismatches.len()
    );
}

//! Гейт ENCODE-пути нейро-кодека HiggsAudioV2 (ref-аудио → коды [8,T]) против
//! Python-эталона. Нужен для voice-cloning.
//!
//! Эталон в `tmp/ov_ref/` (см. `reference/dump_encoder.py`):
//!   enc_input.npy (N,) f32 24k mono — ИДЕНТИЧНЫЙ вход (synaptix ресэмплит сам),
//!   enc_codes.npy (8,T) i64 — model.audio_tokenizer.encode(...).audio_codes[0].
//! Опц. (OV_DUMP_STAGES=1 при дампе): enc_semfeat/e_semantic/e_acoustic/embeddings.
//! Веса — `tmp/ov_unpack/audio_tokenizer/model.safetensors`.
//!
//! Дискретные коды + L2-nearest → единичные флипы из-за f32-дрейфа возможны
//! (особенно HuBERT 12 слоёв). Гейт = % совпадения кодов ≥ 95%. Печатает % +
//! распределение несовпадений по квантизаторам. При наличии stage-дампов —
//! per-stage cos/max-abs для локализации дрейфа.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_omnivoice::audio_encode::CodecEncoder;
use synaptix_tts_omnivoice::config::HiggsAudioConfig;
use synaptix_tts_omnivoice::loader::OmniVoiceCodecWeights;

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

fn load_f32(path: &str) -> (Vec<usize>, Vec<f32>) {
    let (shape, data, descr) = parse_npy(path);
    assert!(descr.contains("f4"), "{path} descr={descr} (expect <f4)");
    let n: usize = shape.iter().product();
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        let o = i * 4;
        v.push(f32::from_le_bytes(data[o..o + 4].try_into().unwrap()));
    }
    (shape, v)
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

fn cos_maxabs(a: &[f32], b: &[f32]) -> (f64, f64) {
    let n = a.len().min(b.len());
    let (mut dot, mut na, mut nb, mut maxabs) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for i in 0..n {
        let (x, y) = (a[i] as f64, b[i] as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        maxabs = maxabs.max((x - y).abs());
    }
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    };
    (cos, maxabs)
}

fn stage_cmp(label: &str, got: &Tensor, ref_path: &str) {
    if !Path::new(ref_path).exists() {
        return;
    }
    let (shape, refv) = load_f32(ref_path);
    let gv: Vec<f32> = got
        .to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    let (cos, maxabs) = cos_maxabs(&gv, &refv);
    eprintln!(
        "[encoder_gate] stage {label}: got {:?} ref {:?} cos={cos:.6} max-abs={maxabs:.4e} \
         (lens got={} ref={})",
        got.dims(),
        shape,
        gv.len(),
        refv.len()
    );
}

#[test]
fn encoder_gate() {
    let weights_path = format!("{UNPACK}/audio_tokenizer/model.safetensors");
    let cfg_path = format!("{UNPACK}/audio_tokenizer/config.json");
    if !Path::new(&weights_path).exists()
        || !Path::new(&format!("{REF}/enc_codes.npy")).exists()
        || !Path::new(&format!("{REF}/enc_input.npy")).exists()
        || !Path::new(&cfg_path).exists()
    {
        eprintln!("[encoder_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let cfg = HiggsAudioConfig::from_json_bytes(&std::fs::read(&cfg_path).unwrap())
        .expect("parse audio_tokenizer config");

    let weights = OmniVoiceCodecWeights::load_safetensors(&weights_path, Device::Cpu, DType::F32)
        .expect("load codec weights");
    eprintln!("[encoder_gate] loaded {} codec tensors", weights.len());

    let enc = CodecEncoder::build(&cfg, &weights).expect("build encoder");
    eprintln!("[encoder_gate] n_q={}", enc.n_q());

    let (in_shape, in_v) = load_f32(&format!("{REF}/enc_input.npy"));
    let n: usize = in_shape.iter().product();
    eprintln!("[encoder_gate] input samples={n} shape={in_shape:?}");
    let input = Tensor::from_vec(in_v, vec![n], Device::Cpu).unwrap();

    let stages = enc.encode_stages(&input).expect("encode");

    // Per-stage cos/max-abs (если дампы stage есть).
    stage_cmp("semantic_features", &stages.semantic_features, &format!("{REF}/enc_semfeat.npy"));
    stage_cmp("e_semantic", &stages.e_semantic, &format!("{REF}/enc_e_semantic.npy"));
    stage_cmp("e_acoustic", &stages.e_acoustic, &format!("{REF}/enc_e_acoustic.npy"));
    stage_cmp("embeddings", &stages.embeddings, &format!("{REF}/enc_embeddings.npy"));

    let (codes_shape, ref_codes) = load_i64(&format!("{REF}/enc_codes.npy"));
    let n_q = codes_shape[0];
    let t = codes_shape[1];
    assert_eq!(stages.codes.dims(), &[n_q, t], "codes shape mismatch");

    let got = stages.codes.flatten_all().unwrap().to_vec1::<i64>().unwrap();
    assert_eq!(got.len(), ref_codes.len());

    let total = got.len();
    let mut matches = 0usize;
    let mut per_q_mismatch = vec![0usize; n_q];
    let mut per_q_total = vec![0usize; n_q];
    for c in 0..n_q {
        for j in 0..t {
            let idx = c * t + j;
            per_q_total[c] += 1;
            if got[idx] == ref_codes[idx] {
                matches += 1;
            } else {
                per_q_mismatch[c] += 1;
            }
        }
    }
    let pct = 100.0 * matches as f64 / total as f64;
    eprintln!("[encoder_gate] OVERALL match {matches}/{total} = {pct:.2}%");
    for c in 0..n_q {
        let q_pct = 100.0 * (per_q_total[c] - per_q_mismatch[c]) as f64 / per_q_total[c] as f64;
        eprintln!(
            "[encoder_gate]   q{c}: {}/{} = {q_pct:.2}% ({} mismatches)",
            per_q_total[c] - per_q_mismatch[c],
            per_q_total[c],
            per_q_mismatch[c]
        );
    }

    assert!(pct >= 95.0, "code match {pct:.2}% < 95%");
}

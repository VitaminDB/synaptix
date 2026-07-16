//! Гейт decode-пути нейро-кодека HiggsAudioV2 против Python-эталона (CPU F32).
//!
//! Эталон в `tmp/ov_ref/` (см. `reference/dump_codec.py`):
//!   gen_codes.npy (8,T i64) → audio_tokenizer.decode → wav_ref.npy (samples f32).
//! Веса — `tmp/ov_unpack/audio_tokenizer/model.safetensors`,
//! конфиг — `.../audio_tokenizer/config.json`.
//!
//! Кросс-фреймворк (conv/snake accumulation, разный порядок редукций) НЕ
//! бит-идентичен → гейт = cosine/корреляция ≥ 0.99 И совпадение формы. Цель
//! cosine ≥ 0.999. Печатает cosine, max-abs, RMS обоих.
//!
//! Гейт guarded: skip (return) если весов/эталона нет на диске.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_omnivoice::audio_codec::CodecDecoder;
use synaptix_tts_omnivoice::config::HiggsAudioConfig;
use synaptix_tts_omnivoice::loader::OmniVoiceCodecWeights;

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";

/// Минимальный npy-парсер: magic(6) + ver(2) + hdr_len + dict-header + raw.
/// Возвращает (shape, raw_data_bytes, descr_string). См. backbone_gate.rs.
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

fn rms(v: &[f32]) -> f64 {
    let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (s / v.len() as f64).sqrt()
}

#[test]
fn codec_gate() {
    let weights_path = format!("{UNPACK}/audio_tokenizer/model.safetensors");
    let cfg_path = format!("{UNPACK}/audio_tokenizer/config.json");
    if !Path::new(&weights_path).exists()
        || !Path::new(&format!("{REF}/gen_codes.npy")).exists()
        || !Path::new(&format!("{REF}/wav_ref.npy")).exists()
        || !Path::new(&cfg_path).exists()
    {
        eprintln!("[codec_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let cfg = HiggsAudioConfig::from_json_bytes(&std::fs::read(&cfg_path).unwrap())
        .expect("parse audio_tokenizer/config.json");

    let (codes_shape, codes_v) = load_i64(&format!("{REF}/gen_codes.npy"));
    assert_eq!(codes_shape.len(), 2, "gen_codes shape {codes_shape:?}");
    let n_q = codes_shape[0];
    let t = codes_shape[1];
    let (ref_shape, ref_wav) = load_f32(&format!("{REF}/wav_ref.npy"));
    let ref_samples: usize = ref_shape.iter().product();
    eprintln!("[codec_gate] codes [{n_q},{t}] → ref wav shape {ref_shape:?} ({ref_samples} samples)");

    let weights = OmniVoiceCodecWeights::load_safetensors(&weights_path, Device::Cpu, DType::F32)
        .expect("load codec weights");
    eprintln!("[codec_gate] loaded {} codec tensors", weights.len());

    let decoder = CodecDecoder::build(&cfg, &weights, n_q).expect("build decoder");

    let codes = Tensor::from_vec(codes_v, vec![n_q, t], Device::Cpu).unwrap();
    let wav = decoder.decode(&codes).expect("decode");
    let got: Vec<f32> = wav.flatten_all().unwrap().to_vec1::<f32>().unwrap();

    eprintln!(
        "[codec_gate] got {} samples (ref {ref_samples}); hop*T = {}",
        got.len(),
        960 * t
    );

    // cosine / max-abs / rms на общей длине (формы должны совпадать).
    let n = got.len().min(ref_wav.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..n {
        let a = got[i] as f64;
        let b = ref_wav[i] as f64;
        dot += a * b;
        na += a * a;
        nb += b * b;
        let d = (got[i] - ref_wav[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    let rms_got = rms(&got);
    let rms_ref = rms(&ref_wav);

    eprintln!(
        "[codec_gate] cosine={cosine:.6} max_abs={max_abs:.6} rms_got={rms_got:.6} rms_ref={rms_ref:.6}"
    );

    assert_eq!(
        got.len(),
        ref_samples,
        "sample count mismatch: got {} ref {}",
        got.len(),
        ref_samples
    );
    assert!(
        cosine >= 0.99,
        "cosine {cosine:.6} < 0.99 (max_abs={max_abs:.6})"
    );
}

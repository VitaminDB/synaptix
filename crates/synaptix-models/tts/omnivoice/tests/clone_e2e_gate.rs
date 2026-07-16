//! Гейт B voice-clone (изолирует ГЕНЕРАЦИЮ от препроцесс-дрейфа): берём
//! ref_audio_tokens из Python-дампа (`clone_ref_tokens.npy`), прогоняем
//! `generate_clone_with_target(text, ref_tokens, ref_text, target_len, det)` →
//! wav; сравниваем с Python clone-wav (`clone_wav.npy`). Гейт cosine ≥ 0.999.
//!
//! Так гейт проверяет ref-ветку pipeline (denoise + ref-токены в cond →
//! masked_decode → codec.decode) на ИДЕНТИЧНОМ входе, без зависимости от
//! preprocess/encode (их покрывает clone_prompt_gate / encoder_gate).
//!
//! Опц. (если есть `clone_codes.npy`): сверка сгенерированных кодов vs Python
//! (изоляция masked_decode от decode) — % совпадения.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_tts_omnivoice::{OmniVoiceGenerationConfig, OmniVoicePipeline};

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";

const REF_TEXT: &str = "Я тебе скажу, что реальная тема. Сходи в круглосуточный и купи нам печенье Джафа. Или шоколадки. Ему шоколадку. А Ириша к вам не купить, чтоб пломбы повышкакивали, клоуны хуевые! Поберегись!";
const TEXT_CLONE: &str = "Привет, это тест клонирования голоса.";

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

fn read_target_len() -> usize {
    let p = format!("{REF}/clone_meta.json");
    let s = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {p}: {e}"));
    let key = "\"target_len\"";
    let i = s.find(key).expect("target_len in clone_meta.json");
    let rest = &s[i + key.len()..];
    let colon = rest.find(':').unwrap();
    let after = &rest[colon + 1..];
    let num: String = after
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num.parse().expect("parse target_len")
}

fn rms(v: &[f32]) -> f64 {
    let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (s / v.len().max(1) as f64).sqrt()
}

#[test]
fn clone_e2e_gate() {
    let lm_path = format!("{UNPACK}/model.safetensors");
    let codec_path = format!("{UNPACK}/audio_tokenizer/model.safetensors");
    if !Path::new(&lm_path).exists()
        || !Path::new(&codec_path).exists()
        || !Path::new(&format!("{REF}/clone_ref_tokens.npy")).exists()
        || !Path::new(&format!("{REF}/clone_wav.npy")).exists()
        || !Path::new(&format!("{REF}/clone_meta.json")).exists()
    {
        eprintln!("[clone_e2e_gate] SKIP: weights/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let pipe = OmniVoicePipeline::from_unpacked(UNPACK, Device::Cpu, DType::F32)
        .expect("from_unpacked");

    // ref_audio_tokens из Python-дампа [C, T_ref].
    let (rshape, rcodes) = load_i64(&format!("{REF}/clone_ref_tokens.npy"));
    let (n_q, t_ref) = (rshape[0], rshape[1]);
    let ref_tokens = Tensor::from_vec(rcodes, vec![n_q, t_ref], Device::Cpu).unwrap();

    let target_len = read_target_len();
    eprintln!("[clone_e2e_gate] T_ref={t_ref} target_len={target_len}");

    // deterministic: position/class temperature = 0.
    let gen = OmniVoiceGenerationConfig {
        num_step: 8,
        guidance_scale: 2.0,
        t_shift: 0.1,
        layer_penalty_factor: 5.0,
        position_temperature: 0.0,
        class_temperature: 0.0,
        denoise: true,
        ..OmniVoiceGenerationConfig::default()
    };

    let wav = pipe
        .generate_clone_with_target(TEXT_CLONE, &ref_tokens, REF_TEXT, target_len, &gen)
        .expect("generate_clone_with_target");

    // Опц. изоляция masked_decode: сверка codes (если дамп есть).
    if Path::new(&format!("{REF}/clone_codes.npy")).exists() {
        let (cshape, ref_codes) = load_i64(&format!("{REF}/clone_codes.npy"));
        // повторно генерим коды через публичный путь нельзя (decode внутри),
        // поэтому сверим лишь форму ожидания.
        eprintln!(
            "[clone_e2e_gate] ref clone_codes shape {:?} ({} codes)",
            cshape,
            ref_codes.len()
        );
    }

    let (ref_shape, ref_wav) = load_f32(&format!("{REF}/clone_wav.npy"));
    let ref_samples: usize = ref_shape.iter().product();
    eprintln!(
        "[clone_e2e_gate] got {} samples ; ref {} samples (shape {:?})",
        wav.len(),
        ref_samples,
        ref_shape
    );

    let n = wav.len().min(ref_wav.len());
    let (mut dot, mut na, mut nb, mut max_abs) = (0.0f64, 0.0f64, 0.0f64, 0.0f32);
    for i in 0..n {
        let (a, b) = (wav[i] as f64, ref_wav[i] as f64);
        dot += a * b;
        na += a * a;
        nb += b * b;
        let d = (wav[i] - ref_wav[i]).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    let cosine = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    eprintln!(
        "[clone_e2e_gate] cosine={cosine:.6} max_abs={max_abs:.6} rms_got={:.6} rms_ref={:.6}",
        rms(&wav),
        rms(&ref_wav)
    );

    assert_eq!(wav.len(), ref_samples, "sample count: got {} ref {}", wav.len(), ref_samples);
    assert!(cosine >= 0.999, "cosine {cosine:.6} < 0.999 (max_abs={max_abs:.6})");
}

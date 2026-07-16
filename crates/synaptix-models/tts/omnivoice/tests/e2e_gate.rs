//! e2e-гейт пайплайна OmniVoice (текст → волна) против Python-эталона.
//!
//! Прогоняет `OmniVoicePipeline::generate_with_target("Hello world.", 12, ...)`
//! (auto-режим, явный target_len=12) и сравнивает wav с
//! `tmp/ov_ref/wav_ref.npy` (11520 f32) — cosine ≥ 0.999.
//! Это та же цепочка inputs→codes→wav, что уже сверена по частям (backbone/
//! decode/codec гейты), но теперь через публичный pipeline из реального текста.
//!
//! Снапшот: `tmp/ov_unpack/` (model.safetensors +
//! audio_tokenizer/model.safetensors + config + tokenizer).
//!
//! Гейт guarded: skip (return) если весов/эталона нет на диске.

use std::path::Path;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_omnivoice::{OmniVoiceGenerationConfig, OmniVoicePipeline};

const UNPACK: &str = "tmp/ov_unpack";
const REF: &str = "tmp/ov_ref";

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

fn rms(v: &[f32]) -> f64 {
    let s: f64 = v.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (s / v.len().max(1) as f64).sqrt()
}

#[test]
fn e2e_gate() {
    let lm_path = format!("{UNPACK}/model.safetensors");
    let codec_path = format!("{UNPACK}/audio_tokenizer/model.safetensors");
    if !Path::new(&lm_path).exists()
        || !Path::new(&codec_path).exists()
        || !Path::new(&format!("{UNPACK}/tokenizer.json")).exists()
        || !Path::new(&format!("{REF}/wav_ref.npy")).exists()
    {
        eprintln!("[e2e_gate] SKIP: weights/tokenizer/ref not on disk");
        return;
    }
    synaptix_kernels_cpu::ensure_registered();

    let pipe = OmniVoicePipeline::from_unpacked(UNPACK, Device::Cpu, DType::F32)
        .expect("from_unpacked");
    eprintln!("[e2e_gate] pipeline built (frame_rate={})", pipe.frame_rate());

    // Gate-параметры из gen_meta.json (num_step=8, g=2.0, t_shift=0.1, lpf=5.0).
    // greedy (position/class temperature=0) для детерминизма.
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
        .generate_with_target("Hello world.", 12, &gen)
        .expect("generate_with_target");

    let (ref_shape, ref_wav) = load_f32(&format!("{REF}/wav_ref.npy"));
    let ref_samples: usize = ref_shape.iter().product();
    eprintln!(
        "[e2e_gate] got {} samples ; ref {} samples (shape {:?})",
        wav.len(),
        ref_samples,
        ref_shape
    );

    let n = wav.len().min(ref_wav.len());
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    for i in 0..n {
        let a = wav[i] as f64;
        let b = ref_wav[i] as f64;
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
        "[e2e_gate] cosine={cosine:.6} max_abs={max_abs:.6} rms_got={:.6} rms_ref={:.6}",
        rms(&wav),
        rms(&ref_wav)
    );

    assert_eq!(wav.len(), ref_samples, "sample count: got {} ref {}", wav.len(), ref_samples);
    assert!(cosine >= 0.999, "cosine {cosine:.6} < 0.999 (max_abs={max_abs:.6})");
}

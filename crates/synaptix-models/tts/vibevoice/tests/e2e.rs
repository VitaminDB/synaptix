use std::path::PathBuf;

use safetensors::SafeTensors;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_vibevoice::config::GenerationConfig;
use synaptix_tts_vibevoice::pipeline::{VibeVoicePipeline, VoiceSample};

fn read_ref(path: &PathBuf) -> Vec<u8> {
    std::fs::read(path).expect("reference safetensors")
}

fn f32_tensor(bytes: &[u8], name: &str) -> Vec<f32> {
    let st = SafeTensors::deserialize(bytes).expect("deserialize");
    let v = st.tensor(name).unwrap_or_else(|_| panic!("tensor {name}"));
    v.data()
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn meta_script(bytes: &[u8]) -> String {
    let (_, meta) = SafeTensors::read_metadata(bytes).expect("read metadata");
    meta.metadata()
        .as_ref()
        .and_then(|m| m.get("script"))
        .cloned()
        .unwrap_or_else(|| "Speaker 1: The quick brown fox jumps over the lazy dog.".to_string())
}

fn i64_tensor(bytes: &[u8], name: &str) -> Vec<i64> {
    let st = SafeTensors::deserialize(bytes).expect("deserialize");
    let v = st.tensor(name).unwrap_or_else(|_| panic!("tensor {name}"));
    v.data()
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    let mut dot = 0f64;
    let mut na = 0f64;
    let mut nb = 0f64;
    for i in 0..n {
        dot += a[i] as f64 * b[i] as f64;
        na += a[i] as f64 * a[i] as f64;
        nb += b[i] as f64 * b[i] as f64;
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-30)
}

#[test]
fn end_to_end_matches_reference() {
    let (Ok(bundle), Ok(refpath)) = (
        std::env::var("SYN_VV_BUNDLE"),
        std::env::var("SYN_VV_GEN_REF"),
    ) else {
        eprintln!("SYN_VV_BUNDLE / SYN_VV_GEN_REF не заданы — тест пропущен");
        return;
    };
    synaptix_kernels_cpu::ensure_registered();
    let device = match std::env::var("SYN_VV_DEVICE").as_deref() {
        Ok("cpu") => Device::Cpu,
        _ => {
            synaptix_kernels_cuda::ensure_registered();
            Device::Cuda(0)
        }
    };

    let bytes = read_ref(&PathBuf::from(refpath));
    let voice = f32_tensor(&bytes, "voice_raw");
    let want_tokens = i64_tensor(&bytes, "generated_tokens");
    let want_audio = f32_tensor(&bytes, "audio");

    let pipeline = VibeVoicePipeline::from_syn(&bundle, device, DType::F32).expect("pipeline");
    let cfg = GenerationConfig {
        cfg_scale: 1.3,
        ddpm_inference_steps: 20,
        zero_noise: true,
        ..GenerationConfig::default()
    };
    let script = meta_script(&bytes);
    println!("script: {script}");
    let out = pipeline
        .synthesize(&script, &[VoiceSample::new(voice, 24_000)], &cfg)
        .expect("synthesize");

    println!(
        "tokens: got {} want {}; audio: got {} want {}",
        out.tokens.len(),
        want_tokens.len(),
        out.audio.len(),
        want_audio.len()
    );
    assert_eq!(
        out.tokens.len(),
        want_tokens.len(),
        "длина последовательности токенов"
    );
    for (i, (a, b)) in out.tokens.iter().zip(want_tokens.iter()).enumerate() {
        assert_eq!(a, b, "токен {i}");
    }
    assert_eq!(out.audio.len(), want_audio.len(), "длина аудио");
    let chunk = 3200usize;
    let n_chunks = out.audio.len() / chunk;
    let mut loud: Vec<f32> = Vec::new();
    let mut loud_ref: Vec<f32> = Vec::new();
    for i in 0..n_chunks {
        let a = &out.audio[i * chunk..(i + 1) * chunk];
        let b = &want_audio[i * chunk..(i + 1) * chunk];
        let rms = (b.iter().map(|v| v * v).sum::<f32>() / chunk as f32).sqrt();
        if rms > 1e-3 {
            loud.extend_from_slice(a);
            loud_ref.extend_from_slice(b);
        }
    }
    let total = cosine(&out.audio, &want_audio);
    println!("audio cos(all)={total:.8}, звучащих кадров {}/{n_chunks}", loud.len() / chunk);
    if loud.len() >= chunk * 4 {
        let c = cosine(&loud, &loud_ref);
        println!("audio cos(loud)={c:.8}");
        assert!(c >= 0.999, "audio cos {c:.8} < 0.999");
    } else {
        println!(
            "эталон почти тишина (zero-noise вырожденный режим) — гейт только по токенам и длине"
        );
    }
    if let Ok(p) = std::env::var("SYN_VV_DUMP") {
        let mut bytes = Vec::with_capacity(out.audio.len() * 4);
        for v in &out.audio {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::write(&p, bytes).expect("dump audio");
        println!("dumped {p}");
    }
}

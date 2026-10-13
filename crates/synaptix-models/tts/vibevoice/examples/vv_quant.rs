//! Качество VibeVoice в кванте: `vv_quant модель.syn bf16|mxfp8|nvfp4 префикс` —
//! три фразы в `префикс_N.wav` (без образца голоса, seed 7, cfg 1.3, 20 шагов).
//! Политика слоёв NVFP4 — `SYN_VV_POLICY`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_tts_vibevoice::{GenerationConfig, VibeVoicePipeline};

const TEXTS: &[&str] = &[
    "Привет! Это проверка качества синтеза речи в разных квантах.",
    "Сегодня тёплый осенний день, и мы идём гулять в парк.",
    "Звёзды бывают очень разными: красными, белыми и голубыми.",
];

fn wav(path: &str, pcm: &[f32], sr: u32) {
    let mut w = Vec::new();
    let n = (pcm.len() * 2) as u32;
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&(36 + n).to_le_bytes());
    w.extend_from_slice(b"WAVEfmt ");
    for v in [16u32.to_le_bytes().to_vec(), 1u16.to_le_bytes().to_vec(), 1u16.to_le_bytes().to_vec(), sr.to_le_bytes().to_vec(), (sr * 2).to_le_bytes().to_vec(), 2u16.to_le_bytes().to_vec(), 16u16.to_le_bytes().to_vec()] {
        w.extend_from_slice(&v);
    }
    w.extend_from_slice(b"data");
    w.extend_from_slice(&n.to_le_bytes());
    for x in pcm {
        w.extend_from_slice(&((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes());
    }
    std::fs::write(path, w).unwrap();
}

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let a: Vec<String> = std::env::args().skip(1).collect();
    let q = match a[1].as_str() {
        "nvfp4" => Some(DType::NVFP4),
        "mxfp8" => Some(DType::MXFP8),
        _ => None,
    };
    let p = VibeVoicePipeline::from_syn_quant(&a[0], Device::Cuda(0), DType::BF16, q).unwrap();
    let cfg = GenerationConfig { cfg_scale: 1.3, ddpm_inference_steps: 20, seed: 7, ..Default::default() };
    for (k, t) in TEXTS.iter().enumerate() {
        let out = p.synthesize(&format!("Speaker 0: {t}"), &[], &cfg).unwrap();
        wav(&format!("{}_{k}.wav", a[2]), &out.audio, p.sample_rate());
    }
}

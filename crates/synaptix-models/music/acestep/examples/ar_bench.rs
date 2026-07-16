//! Изолированный AR (5Hz LM) bench. `AR_DUR` сек (180≈900 кодов), `AR_CFG`.
use std::path::Path;
use std::time::Instant;

use synaptix_core::dtype::DType;
use synaptix_core::device::Device;
use synaptix_music_acestep::ar::{ar_generate, CodesGenOptions};
use synaptix_music_acestep::lm::AceStepLm;
use synaptix_music_acestep::loader::read_bundle_file;
use synaptix_music_acestep::tokenizer::{AceTokenizer, Metadata};

fn main() {
    synaptix_kernels_cpu::ensure_registered();
    synaptix_kernels_cuda::ensure_registered();
    let device = Device::Cuda(0);
    let lm_path = Path::new("storage/syn_models/acestep_5hz_lm_1.7b.syn");
    let getv = |k: &str, d: usize| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let dur = getv("AR_DUR", 180);
    let cfg: f32 = std::env::var("AR_CFG").ok().and_then(|s| s.parse().ok()).unwrap_or(2.0);
    let n = getv("AR_N", 2);

    let lm = AceStepLm::open(lm_path, device, DType::BF16, DType::BF16, dur * 5 + 2048).expect("lm");
    let tok = AceTokenizer::from_bytes(&read_bundle_file(lm_path, "tokenizer.json").unwrap()).unwrap();
    let caption = "a calm lo-fi hip hop beat with mellow piano and soft drums";
    let base = Metadata { caption: caption.into(), duration: dur as u32, ..Metadata::default() };
    let temp: f32 = std::env::var("AR_TEMP").ok().and_then(|s| s.parse().ok()).unwrap_or(0.85);
    let opts = CodesGenOptions { cfg_scale: cfg, seed: 42, temperature: temp, ..Default::default() };

    for i in 0..n {
        let t0 = Instant::now();
        let (codes, _m) = ar_generate(&lm, &tok, caption, "", &base, &opts, false).expect("ar");
        let dt = t0.elapsed().as_secs_f32();
        let sum: u64 = codes.iter().map(|&c| c as u64).sum();
        let head: Vec<u32> = codes.iter().take(8).copied().collect();
        let tail: Vec<u32> = codes.iter().rev().take(4).rev().copied().collect();
        eprintln!("[ar-bench] run {i}: {:.2}s ({} codes, {:.0} tok/s, cfg={cfg}) sum={sum} head={head:?} tail={tail:?}", dt, codes.len(), codes.len() as f32 / dt);
    }
}

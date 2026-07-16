use std::path::PathBuf;

use synaptix_core::{device::Device, dtype::DType};
use synaptix_music_acestep::ar::{generate_codes, CodesGenOptions};
use synaptix_music_acestep::lm::AceStepLm;
use synaptix_music_acestep::loader::read_bundle_file;
use synaptix_music_acestep::tokenizer::{AceTokenizer, Metadata};

fn lm_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_5hz_lm_1.7b.syn");
    p.exists().then_some(p)
}

#[test]
fn generate_codes_greedy() {
    if std::env::var("ACESTEP_GENERATE").is_err() {
        return;
    }
    let Some(path) = lm_path() else { return };
    synaptix_kernels_cpu::ensure_registered();
    let bytes = read_bundle_file(&path, "tokenizer.json").expect("tokenizer.json");
    let tok = AceTokenizer::from_bytes(&bytes).expect("tokenizer");
    let lm = AceStepLm::open(&path, Device::Cpu, DType::BF16, DType::BF16, 256).expect("lm");

    let meta = Metadata { caption: "calm ambient piano".into(), duration: 2, ..Metadata::default() };
    let opts = CodesGenOptions::default(); // greedy
    let codes = generate_codes(&lm, &tok, "calm ambient piano", "", &meta, &opts).expect("gen");
    eprintln!("[acestep-ar] {} codes: {:?}", codes.len(), &codes[..codes.len().min(10)]);
    assert_eq!(codes.len(), 10);
    assert!(codes.iter().all(|&c| c < 64000));
}

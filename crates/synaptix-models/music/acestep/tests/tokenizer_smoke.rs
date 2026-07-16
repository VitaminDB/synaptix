use std::path::PathBuf;

use synaptix_music_acestep::loader::read_bundle_file;
use synaptix_music_acestep::tokenizer::{AceTokenizer, Metadata, NUM_AUDIO_CODES};

fn lm_path() -> Option<PathBuf> {
    let p = PathBuf::from("storage/syn_models/acestep_5hz_lm_1.7b.syn");
    p.exists().then_some(p)
}

#[test]
fn audio_code_mapping_contiguous() {
    let Some(path) = lm_path() else { return };
    let bytes = read_bundle_file(&path, "tokenizer.json").expect("tokenizer.json");
    let tok = AceTokenizer::from_bytes(&bytes).expect("tokenizer");

    let base = tok.audio_base();
    eprintln!("[acestep-tok] audio_base={base} eos={} bos={}", tok.eos(), tok.bos());
    assert_eq!(tok.code_to_id(0), base);
    assert_eq!(tok.code_to_id(63999), base + 63999);

    // round-trip code <-> id
    for &n in &[0u32, 1, 5, 100, 63999] {
        assert_eq!(tok.id_to_code(tok.code_to_id(n)), Some(n));
        let txt = tok.decode(&[tok.code_to_id(n)]).expect("decode");
        assert_eq!(txt, format!("<|audio_code_{n}|>"), "id->text mismatch for code {n}");
    }
    // id just outside range is not a code
    assert_eq!(tok.id_to_code(base + NUM_AUDIO_CODES), None);
}

#[test]
fn codes_prompt_encodes() {
    let Some(path) = lm_path() else { return };
    let bytes = read_bundle_file(&path, "tokenizer.json").expect("tokenizer.json");
    let tok = AceTokenizer::from_bytes(&bytes).expect("tokenizer");
    let meta = Metadata { caption: "calm piano".into(), duration: 10, ..Metadata::default() };
    let prompt = tok.build_codes_prompt("calm piano", "", &meta);
    let ids = tok.encode(&prompt).expect("encode");
    eprintln!("[acestep-tok] codes prompt -> {} tokens", ids.len());
    assert!(!ids.is_empty());
    // <|im_start|> must tokenize as a known special id (not split)
    let im_start = tok.encode("<|im_start|>").expect("enc im_start");
    assert_eq!(im_start.len(), 1, "<|im_start|> should be a single token");
}

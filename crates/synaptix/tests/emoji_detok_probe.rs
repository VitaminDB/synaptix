//! Регрессия к «эмодзи пропадают в стриме»: байтовый BPE режет 4-байтовый
//! эмодзи на несколько токенов, и decode частичной последовательности даёт
//! «�» — стрим обязан придерживать такой хвост (см. `stream_delta` в
//! facade::llm), иначе дельта с «�» уходит в UI, а настоящий символ теряется.
//! Тест гейтится наличием бандла qwen3.8 (55 ГБ не таскаем в CI).
use std::path::Path;
use synaptix::facade::arch::read_model_file;
use synaptix_tokenizer::{HfTokenizer, Tokenizer};

#[test]
fn multi_token_emoji_partial_decode_yields_replacement_char() {
    let bundle = Path::new("/run/media/storage/syn_models/qwen3.8-27b.syn");
    if !bundle.exists() {
        eprintln!("bundle отсутствует — скип");
        return;
    }
    let bytes = read_model_file(bundle, "tokenizer.json").expect("tokenizer.json");
    let tok = HfTokenizer::from_bytes(&bytes).expect("parse tokenizer");

    let ids = tok.encode("🎉", false).expect("encode").ids.clone();
    assert!(
        ids.len() > 1,
        "🎉 ожидался многотокенным (иначе регрессия не воспроизводится): {ids:?}"
    );
    for n in 1..ids.len() {
        let partial = tok.decode(&ids[..n], true).expect("decode");
        assert!(
            partial.ends_with('\u{FFFD}'),
            "частичный decode[..{n}] должен давать «�»-хвост, получили {partial:?}"
        );
    }
    assert_eq!(tok.decode(&ids, true).expect("decode"), "🎉");
}

//! Английское произношение (G2P Leokani) и его русская запись для Silero.
//! en_phonemes <leokani.syn> "слова …"
fn main() {
    let a: Vec<String> = std::env::args().collect();
    synaptix_kernels_cpu::ensure_registered();
    let g = synaptix_tts_leokani::g2p::LeokaniG2P::from_syn(std::path::Path::new(&a[1])).expect("g2p");
    for w in a[2].split_whitespace() {
        let ph = g.phonemize(w);
        println!("{w}\t{ph}\t{}", synaptix_tts_silero::english::ipa_to_cyrillic(&ph));
    }
}

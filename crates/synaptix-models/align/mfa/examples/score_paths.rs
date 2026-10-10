//! Отладка: стоимость пути при свободном выборе вариантов и при заданных.
//!   score_paths <model.mfa> <wav> "<текст>" "<фонемы через | по словам>"
use synaptix_align_mfa::{align_alts, AlignOptions, AlignWord, AlignWordAlts, MfaModel};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = MfaModel::load(std::path::Path::new(&a[1])).unwrap();
    let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(&a[2]).unwrap();
    let pcm = if sr == 16000 { pcm } else { synaptix_audio::resample(&pcm, sr, 16000).unwrap() };
    let feats = model.features(&pcm);
    let words: Vec<&str> = a[3].split_whitespace().collect();
    let free: Vec<AlignWordAlts> = words
        .iter()
        .map(|w| AlignWordAlts { alts: model.lexicon.get(w).unwrap().iter().map(|p| (AlignWord::from_pron(p), p.prob)).collect() })
        .collect();
    let forced: Vec<AlignWordAlts> = words
        .iter()
        .zip(a[4].split('|'))
        .map(|(w, ph)| {
            let ph: Vec<String> = ph.split_whitespace().map(String::from).collect();
            let p = model.lexicon.get(w).unwrap().iter().find(|p| p.phones == ph).expect("вариант");
            AlignWordAlts { alts: vec![(AlignWord::from_pron(p), p.prob)] }
        })
        .collect();
    for (name, ws) in [("свободно", free), ("как MFA", forced)] {
        let al = align_alts(&model, &feats, &ws, &AlignOptions::default()).unwrap();
        let ph: Vec<&str> = al.phones.iter().map(|p| p.phone.as_str()).collect();
        println!("{name}: score {:.3}  {}", al.score, ph.join(" "));
    }
}

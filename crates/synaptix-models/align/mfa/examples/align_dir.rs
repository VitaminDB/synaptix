//! Выровнять каталог `*.wav` + `*.lab` (как корпус MFA): JSON на stdout.
//!   align_dir <model.mfa> <каталог>
use synaptix_align_mfa::{align_alts, AlignOptions, AlignWord, AlignWordAlts, MfaModel};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = MfaModel::load(std::path::Path::new(&a[1])).expect("model");
    let mut out = serde_json::Map::new();
    let mut entries: Vec<_> = std::fs::read_dir(&a[2]).unwrap().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "wav")).collect();
    entries.sort();
    for wav in entries {
        let lab = std::fs::read_to_string(wav.with_extension("lab")).unwrap();
        let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(&wav).unwrap();
        let pcm = if sr == 16000 { pcm } else { synaptix_audio::resample(&pcm, sr, 16000).unwrap() };
        let feats = model.features(&pcm);
        let words: Option<Vec<AlignWordAlts>> = lab
            .split_whitespace()
            .map(|w| {
                model.lexicon.get(w).map(|ps| AlignWordAlts {
                    alts: ps.iter().map(|p| (AlignWord::from_pron(p), p.prob)).collect(),
                })
            })
            .collect();
        let Some(words) = words else { continue };
        if let Ok(al) = align_alts(&model, &feats, &words, &AlignOptions::default()) {
            let spans: Vec<_> = al
                .phones
                .iter()
                .map(|p| serde_json::json!([p.phone, p.start as f32 * al.frame_shift, p.end as f32 * al.frame_shift]))
                .collect();
            out.insert(wav.file_stem().unwrap().to_string_lossy().into(), spans.into());
        }
    }
    println!("{}", serde_json::Value::Object(out));
}

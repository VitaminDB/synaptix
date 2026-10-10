//! Гейт выравнивателя MFA против Kaldi (`reference/make_ref.sh` → `tmp/mfa_ref`):
//! MFCC, признаки после CMVN/splice/LDA, лог-правдоподобия GMM; затем
//! выравнивание фразы. Без эталона — SKIP.

use std::path::PathBuf;

use synaptix_align_mfa::{align, features::Mfcc, AlignOptions, AlignWord, MfaModel};

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../..")
}

fn refd() -> PathBuf {
    root().join("tmp/mfa_ref")
}

fn mfa_model_dir() -> PathBuf {
    root().join("../synspeak/tools/build/mfa/eng/english_us_arpa")
}

/// Kaldi `ark,t` с одной матрицей: строки чисел между `[` и `]`.
fn kaldi_matrix(path: &PathBuf) -> (usize, Vec<f32>) {
    let src = std::fs::read_to_string(path).unwrap();
    let mut rows = 0;
    let mut v = Vec::new();
    for line in src.lines() {
        let nums: Vec<f32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
        if !nums.is_empty() {
            rows += 1;
            v.extend(nums);
        }
    }
    (rows, v)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs() / (1.0 + y.abs())).fold(0.0, f32::max)
}

#[test]
fn mfa_gate() {
    if !refd().join("likes.txt").exists() || !mfa_model_dir().join("meta.json").exists() {
        eprintln!("[mfa_gate] SKIP");
        return;
    }
    let read = |p: PathBuf| std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    let dict = root().join("../synspeak/tools/build/mfa/mfa_root/pretrained_models/dictionary/english_us_arpa.dict");
    let model = MfaModel::from_kaldi_text(
        &read(refd().join("final.alimdl.txt")),
        &read(refd().join("tree.txt")),
        &read(refd().join("lda.txt")),
        &read(mfa_model_dir().join("meta.json")),
        &read(dict),
    )
    .unwrap();
    let bin = refd().join("model.mfa");
    model.save(&bin).unwrap();
    let model = MfaModel::load(&bin).unwrap();
    eprintln!("[mfa_gate] pdfs {}, гауссиан {}, слов {}", model.gmms.num_pdfs(), model.gmms.gconsts.len(), model.lexicon.words.len());

    let (pcm, sr) = synaptix_audio::io::read_wav_mono_f32(root().join("tmp/tts/en1_16k.wav")).unwrap();
    assert_eq!(sr, 16000);
    let scaled: Vec<f32> = pcm.iter().map(|x| x * 32768.0).collect();
    let mfcc = Mfcc::new(model.features.clone()).compute(&scaled);
    let (rows, kaldi) = kaldi_matrix(&refd().join("mfcc.txt"));
    assert_eq!(mfcc.len(), kaldi.len(), "кадров: наши {} Kaldi {rows}", mfcc.len() / 13);
    let e = max_rel(&mfcc, &kaldi);
    eprintln!("[mfa_gate] MFCC {rows} кадров, макс. отн. ошибка {e:.2e}");
    assert!(e < 2e-3, "MFCC расходится: {e}");

    let feats = model.features(&pcm);
    let (_, kf) = kaldi_matrix(&refd().join("feats.txt"));
    let e = max_rel(&feats, &kf);
    eprintln!("[mfa_gate] признаки LDA: макс. отн. ошибка {e:.2e}");
    assert!(e < 5e-3, "признаки расходятся: {e}");

    let (lr, likes) = kaldi_matrix(&refd().join("likes.txt"));
    let np = likes.len() / lr;
    let mut worst = 0f32;
    for t in (0..lr).step_by(17) {
        for p in (0..np).step_by(97) {
            let ours = model.gmms.loglike(p, &kf[t * 40..(t + 1) * 40]);
            worst = worst.max((ours - likes[t * np + p]).abs() / (1.0 + likes[t * np + p].abs()));
        }
    }
    eprintln!("[mfa_gate] лог-правдоподобия: макс. отн. ошибка {worst:.2e}");
    assert!(worst < 1e-3, "GMM расходится: {worst}");

    let text = "the weather is really nice today so we decided to walk to the park";
    let words: Vec<AlignWord> = text
        .split_whitespace()
        .map(|w| {
            let p = &model.lexicon.get(w).unwrap()[0];
            AlignWord::from_pron(p)
        })
        .collect();
    let t0 = std::time::Instant::now();
    let a = align(&model, &feats, &words, &AlignOptions::default()).unwrap();
    eprintln!("[mfa_gate] выравнивание за {:?}", t0.elapsed());
    for p in &a.phones {
        eprintln!("  {:>5.2}–{:>5.2}  {:<4} {:?}", p.start as f32 * a.frame_shift, p.end as f32 * a.frame_shift, p.phone, p.word);
    }
}

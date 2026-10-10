//! Отладка: акустика покадрового пути Kaldi (pdf по кадрам) против нашего.
//!   path_debug <model.mfa> <feats.txt> <pdf.txt> "<текст>"
use synaptix_align_mfa::{align_alts, AlignOptions, AlignWord, AlignWordAlts, MfaModel};

fn nums(path: &str) -> Vec<f32> {
    std::fs::read_to_string(path).unwrap().split_whitespace().filter_map(|t| t.parse().ok()).collect()
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let model = MfaModel::load(std::path::Path::new(&a[1])).unwrap();
    let feats = nums(&a[2]);
    let pdfs: Vec<usize> = nums(&a[3]).into_iter().map(|x| x as usize).collect();
    let t = feats.len() / 40;
    let kaldi: f32 = (0..t).map(|i| model.gmms.loglike(pdfs[i], &feats[i * 40..(i + 1) * 40])).sum();
    println!("кадров {t} (pdf {}), акустика пути Kaldi {kaldi:.1} ({:.4}/кадр)", pdfs.len(), kaldi / t as f32);
    let words: Vec<AlignWordAlts> = a[4]
        .split_whitespace()
        .map(|w| AlignWordAlts { alts: model.lexicon.get(w).unwrap().iter().map(|p| (AlignWord::from_pron(p), p.prob)).collect() })
        .collect();
    let mut o = AlignOptions::default();
    o.beam = None;
    let al = align_alts(&model, &feats, &words, &o).unwrap();
    let ours: f32 = al.phones.iter().map(|p| p.loglike * (p.end - p.start) as f32).sum();
    println!("наш путь: score {:.2}, акустика {ours:.1}: {}", al.score, al.phones.iter().map(|p| p.phone.as_str()).collect::<Vec<_>>().join(" "));
}

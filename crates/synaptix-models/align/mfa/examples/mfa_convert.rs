//! Модель MFA (текстовые формы Kaldi + meta.json + словарь) → компактный `.mfa`.
//!   mfa_convert <final.mdl.txt> <tree.txt> <lda.txt> <meta.json> <словарь.dict> <out.mfa>
fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 7 {
        eprintln!("mfa_convert <final.mdl.txt> <tree.txt> <lda.txt> <meta.json> <словарь.dict> <out.mfa>");
        std::process::exit(2);
    }
    let r = |i: usize| std::fs::read_to_string(&a[i]).unwrap_or_else(|e| panic!("{}: {e}", a[i]));
    let m = synaptix_align_mfa::MfaModel::from_kaldi_text(&r(1), &r(2), &r(3), &r(4), &r(5)).expect("модель");
    m.save(std::path::Path::new(&a[6])).expect("запись");
    eprintln!("готово: {} pdf, {} гауссиан, {} слов", m.gmms.num_pdfs(), m.gmms.gconsts.len(), m.lexicon.words.len());
}

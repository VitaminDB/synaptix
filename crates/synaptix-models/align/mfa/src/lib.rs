//! Порт выравнивателя Montreal Forced Aligner (Kaldi GMM-HMM) на Rust.
//!
//! Инференс целиком без Kaldi/Python: MFCC бит-в-бит как `compute-mfcc-feats`,
//! CMVN, сплайсинг и LDA+MLLT, диагональные GMM связанных состояний трифонов,
//! дерево контекстов (EventMap), HMM-топология и Витерби по графу фразы с
//! необязательными паузами между словами — как первый проход `mfa align`
//! (модель без адаптации к диктору, `final.alimdl`).
//!
//! Модель грузится из текстовых форм Kaldi (`gmm-copy --binary=false`,
//! `copy-tree --binary=false`, `copy-matrix --binary=false`) и сохраняется в
//! компактный бинарный формат (`MfaModel::save`/`load`).

pub mod align;
pub mod features;
pub mod gmm;
pub mod kaldi_text;
pub mod lexicon;
pub mod model;
pub mod tree;

pub use align::{align, align_alts, AlignOptions, AlignWord, AlignWordAlts, Alignment, PhoneSpan, WordSpan};
pub use features::{FeatureConfig, Mfcc};
pub use lexicon::Lexicon;
pub use model::MfaModel;

#[derive(Debug, thiserror::Error)]
pub enum MfaError {
    #[error("формат: {0}")]
    Format(String),
    #[error("io {0}: {1}")]
    Io(String, std::io::Error),
    #[error("выравнивание: {0}")]
    Align(String),
}

pub type Result<T> = std::result::Result<T, MfaError>;

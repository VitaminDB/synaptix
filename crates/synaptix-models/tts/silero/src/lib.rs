//! Silero TTS v5 (snakers4/silero-models): русские v5_ru…v5_6_ru (5 голосов, ударения и ё —
//! n-граммный расстановщик, омографы — BERT; v5_4+ — интонация вопросов, v5_6 — эмфаза), пакеты СНГ
//! (v5_cis_base/_nostress/_ext: 20+ языков, в т. ч. казахский), кавказские и тюркские. Сеть одна:
//! FastPitch-подобная (длительность, высота тона, «песочные часы») + Vocos-вокодер 48 кГц, PQMF
//! даёт 24 и 8 кГц.

pub mod accentor;
pub mod config;
pub mod frontend;
pub mod homosolver;
pub mod model;
pub mod numbers;
pub mod pipeline;
pub mod translit;

pub use pipeline::{Options, SileroPipeline};

#[derive(Debug, thiserror::Error)]
pub enum SileroError {
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, SileroError>;

/// Частоты, которые даёт вокодер (48 кГц — родная, 24 и 8 — PQMF).
pub const SAMPLE_RATES: [u32; 3] = [48_000, 24_000, 8_000];

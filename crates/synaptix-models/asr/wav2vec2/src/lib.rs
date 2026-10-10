//! Wav2Vec2 / XLS-R + CTC-голова на примитивах synaptix.
//!
//! Архитектура (источник истины — `transformers.models.wav2vec2.modeling_wav2vec2`):
//! нормализованный PCM 16 кГц → свёрточный feature extractor (7 conv1d, LayerNorm
//! или GroupNorm, GELU; шаг 320 сэмплов = 20 мс) → feature projection (LayerNorm +
//! Linear) → позиционная свёртка (grouped conv1d с weight-norm) → N слоёв
//! трансформера (pre-LN при `do_stable_layer_norm`, иначе post-LN) → `lm_head`
//! → log-softmax по словарю CTC.
//!
//! Основной пользователь — оценка произношения: фонемная модель
//! `wav2vec2-xlsr-53-espeak-cv-ft` выдаёт покадровые вероятности IPA-фонем.

pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod train;
pub mod vocab;

pub use config::Wav2Vec2Config;
pub use loader::Wav2Vec2Weights;
pub use model::Wav2Vec2Model;
pub use pipeline::{collapse_ctc, normalize_audio, Wav2Vec2Ctc};
pub use vocab::CtcVocab;

pub type Result<T> = std::result::Result<T, Wav2Vec2Error>;

#[derive(Debug, thiserror::Error)]
pub enum Wav2Vec2Error {
    #[error("config: {0}")]
    Config(String),
    #[error("bundle: {0}")]
    Bundle(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("vocab: {0}")]
    Vocab(String),
    #[error("audio: {0}")]
    Audio(String),
}

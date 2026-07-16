//! Whisper v1 / v2 / v3 — нативная реализация на примитивах synaptix.
//!
//! Энкодер-декодер ASR: conv-stem + 32 pre-LN self-attn слоя энкодера, 4 слоя
//! декодера (turbo) с self-attn + cross-attn. Веса грузятся из `.syn`-бандла в
//! HF-раскладке. Особенности Whisper, учтённые здесь: `k_proj` без bias
//! (q/v/out — с bias), точный erf-`gelu`, обучаемые позиционные эмбеддинги,
//! tied lm_head (= `embed_tokens`).

pub mod config;
pub mod loader;
pub mod mel;
pub mod model;
pub mod pipeline;

pub use config::{GenerationConfig, WhisperConfig};
pub use loader::WhisperWeights;
pub use model::{WhisperDecoder, WhisperEncoder, WhisperModel};
pub use pipeline::{Task, TsSegment, WhisperPipeline};

pub type Result<T> = std::result::Result<T, WhisperError>;

#[derive(Debug, thiserror::Error)]
pub enum WhisperError {
    #[error("config: {0}")]
    Config(String),
    #[error("bundle: {0}")]
    Bundle(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("audio: {0}")]
    Audio(String),
}

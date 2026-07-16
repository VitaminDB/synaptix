//! GigaAM-v3-e2e-CTC — нативный русский Conformer-ASR на примитивах synaptix.
//!
//! Архитектура (источник `~/Temp/GigaAM/gigaam/`): log-mel (torchaudio htk) →
//! StridingSubsampling (conv1d ×2, stride-2) → 16 Conformer-слоёв (Macaron-FFN,
//! RoPE-attention, ConformerConvolution: GLU + depthwise + LayerNorm) → CTC-head
//! (conv1d 1×1) → greedy-CTC → SentencePiece-декод.

pub mod config;
pub mod loader;
pub mod mel;
pub mod model;
pub mod pipeline;
pub mod spm;

pub use config::GigaAmConfig;
pub use loader::GigaAmWeights;
pub use model::GigaAmModel;
pub use pipeline::GigaAm;
pub use spm::SpmDecoder;

pub type Result<T> = std::result::Result<T, GigaAmError>;

#[derive(Debug, thiserror::Error)]
pub enum GigaAmError {
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

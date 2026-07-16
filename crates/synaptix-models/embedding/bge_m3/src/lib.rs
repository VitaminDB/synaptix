//! BGE-M3 — мультиязычный эмбеддер (XLM-RoBERTa backbone) на примитивах synaptix.
//!
//! Нативный порт по upstream-спеке. Реализован dense-эмбеддинг: encoder
//! (24 post-LN BERT-слоя, абсолютные позиции, exact-gelu) → CLS-pool → L2-norm.
//! Multi-vector (ColBERT) и sparse (lexical) головы BGE-M3 — не портированы (для
//! KB нужен только dense).

pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod reranker;

pub use config::BgeConfig;
pub use model::{BgeEncoder, l2_normalize};
pub use pipeline::BgeM3;
pub use reranker::BgeReranker;

pub type Result<T> = std::result::Result<T, BgeError>;

#[derive(Debug, thiserror::Error)]
pub enum BgeError {
    #[error("config: {0}")]
    Config(String),
    #[error("bundle: {0}")]
    Bundle(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("inference: {0}")]
    Inference(String),
}


pub mod ar;
pub mod cond_encoder;
pub mod config;
pub mod dcw;
pub mod detokenizer;
pub mod dit;
pub mod encoder;
pub mod fsq;
pub mod lm;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod scheduler;
pub mod text_encoder;
pub mod tokenizer;
pub mod vae;

use synaptix_core::error::SynaptixError;

#[derive(Debug, thiserror::Error)]
pub enum AceError {
    #[error("config: {0}")]
    Config(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, AceError>;

pub use config::{DitConfig, LmConfig, VaeConfig};
pub use lm::AceStepLm;
pub use vae::AceStepVae;

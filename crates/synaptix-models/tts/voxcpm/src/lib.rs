pub mod audio_io;
pub mod audiovae;
pub mod cfm;
pub mod config;
pub mod fsq;
pub mod loader;
pub mod locdit;
pub mod locenc;
pub mod minicpm;
pub mod model;
pub mod pipeline;
pub mod tokenizer;

pub use config::VoxConfig;
pub use loader::VoxCheckpoint;
pub use model::VoxCpmModel;
pub use pipeline::{GenerateOptions, VoxCpmPipeline, Waveform};

use synaptix_core::error::SynaptixError;

#[derive(Debug, thiserror::Error)]
pub enum VoxError {
    #[error("config: {0}")]
    Config(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] SynaptixError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, VoxError>;

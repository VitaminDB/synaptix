use synaptix_core::error::SynaptixError;
use synaptix_llm_common::ModelError;

#[derive(Debug, thiserror::Error)]
pub enum TrainError {
    #[error("{0}")]
    Core(#[from] SynaptixError),
    #[error("{0}")]
    Model(#[from] ModelError),
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, TrainError>;

pub(crate) fn other(msg: impl Into<String>) -> TrainError {
    TrainError::Other(msg.into())
}

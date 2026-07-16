use thiserror::Error;

#[derive(Debug, Error)]
pub enum DispatchError {
    #[error("no implementation: {0}")]
    NoImpl(String),
    #[error("core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DispatchError>;

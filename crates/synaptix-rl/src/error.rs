use thiserror::Error;

#[derive(Debug, Error)]
pub enum RlError {
    #[error("{0}")] Other(String),
}

pub type Result<T> = std::result::Result<T, RlError>;

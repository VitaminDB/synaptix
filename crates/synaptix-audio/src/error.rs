use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, AudioError>;

#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("io error at {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },

    #[error("io: {0}")]
    IoBare(#[from] std::io::Error),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("wav decode/encode: {0}")]
    Wav(#[from] hound::Error),

    #[error("rubato resampler: {0}")]
    Rubato(String),

    #[error("{0}")]
    Other(String),
}

impl AudioError {
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn other(msg: impl Into<String>) -> Self {
        Self::Other(msg.into())
    }
}

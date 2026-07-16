use thiserror::Error;

#[derive(Debug, Error)]
pub enum IoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),
    #[error("safetensors: {0}")]
    Safetensors(String),
    #[error("bundle: {0}")]
    Bundle(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error("image: {0}")]
    Image(String),
    #[error("video: {0}")]
    Video(String),
    #[error("data: {0}")]
    Data(String),
    #[error("document: {0}")]
    Document(String),
    #[error("{0}")]
    Other(String),
}

impl From<synaptix_bundle::error::Error> for IoError {
    fn from(e: synaptix_bundle::error::Error) -> Self {
        Self::Bundle(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, IoError>;

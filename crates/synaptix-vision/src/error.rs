use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, VisionError>;

#[derive(Debug, thiserror::Error)]
pub enum VisionError {
    #[error("io at {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },

    #[error("io: {0}")]
    IoBare(#[from] std::io::Error),

    #[error("image: {0}")]
    Image(#[from] image::ImageError),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("synaptix-core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),

    #[error("{0}")]
    Other(String),
}

impl VisionError {
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }
}

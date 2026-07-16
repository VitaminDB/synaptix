pub type Result<T> = std::result::Result<T, MultimodalError>;

#[derive(Debug, thiserror::Error)]
pub enum MultimodalError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    #[error("synaptix-core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),

    #[error("{0}")]
    Other(String),
}

impl MultimodalError {
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }

    pub fn shape(msg: impl Into<String>) -> Self {
        Self::ShapeMismatch(msg.into())
    }
}

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, DebugError>;

#[derive(Debug, thiserror::Error)]
pub enum DebugError {
    #[error("io error at {path}: {source}")]
    Io { path: PathBuf, #[source] source: std::io::Error },

    #[error("io: {0}")]
    IoBare(#[from] std::io::Error),

    #[error("invalid dump magic: expected `{expected:?}`, got `{got:?}`")]
    InvalidMagic { expected: [u8; 8], got: [u8; 8] },

    #[error("unsupported dump version: {0}")]
    UnsupportedVersion(u32),

    #[error("unknown dtype tag {0}")]
    UnknownDtype(u32),

    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },

    #[error("dtype mismatch: expected {expected:?}, got {got:?}")]
    DTypeMismatch { expected: synaptix_core::dtype::DType, got: synaptix_core::dtype::DType },

    #[error("non-finite value at position {position}: {kind}")]
    NonFinite { position: usize, kind: &'static str },

    #[error("synaptix-core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),

    #[error("{0}")]
    Other(String),
}

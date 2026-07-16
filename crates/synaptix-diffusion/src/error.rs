pub type Result<T> = std::result::Result<T, DiffusionError>;

#[derive(Debug, thiserror::Error)]
pub enum DiffusionError {
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("step index {idx} out of range (n_steps={n_steps})")]
    StepOutOfRange { idx: usize, n_steps: usize },

    #[error("synaptix-core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),

    #[error("{0}")]
    Other(String),
}

impl DiffusionError {
    pub fn invalid_arg(msg: impl Into<String>) -> Self {
        Self::InvalidArgument(msg.into())
    }
}

use thiserror::Error;

#[derive(Debug, Error)]
pub enum DistError {
    #[error("not initialized")]
    NotInitialized,
    #[error("rank {rank} out of world size {world_size}")]
    RankOutOfRange { rank: usize, world_size: usize },
    #[error("core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, DistError>;

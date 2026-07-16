use thiserror::Error;

#[derive(Debug, Error)]
pub enum InferError {
    #[error("core: {0}")]
    Core(#[from] synaptix_core::error::SynaptixError),
    #[error("kv cache: {0}")]
    KvCache(String),
    #[error("sampling: {0}")]
    Sampling(String),
    #[error("scheduler: {0}")]
    Scheduler(String),
    #[error("session {id}: {msg}")]
    Session { id: u64, msg: String },
    #[error("out of memory: {0}")]
    Oom(String),
    #[error("request cancelled")]
    Cancelled,
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, InferError>;

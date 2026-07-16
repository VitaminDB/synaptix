use thiserror::Error;

#[derive(Debug, Error)]
pub enum CompileError {
    #[error("ir: {0}")]
    Ir(String),
    #[error("codegen: {0}")]
    Codegen(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, CompileError>;

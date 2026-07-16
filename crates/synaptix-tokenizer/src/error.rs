use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, TokenizerError>;

#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    #[error("io error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("io error: {0}")]
    IoBare(#[from] std::io::Error),

    #[error("tokenizer backend error: {0}")]
    Backend(String),

    #[error("invalid tokenizer file `{path}`: {message}")]
    InvalidFile { path: PathBuf, message: String },

    #[error("missing file: {0}")]
    MissingFile(PathBuf),

    #[error("token `{token}` not found in vocabulary")]
    UnknownToken { token: String },

    #[error("id {id} out of vocabulary range")]
    UnknownId { id: u32 },

    #[error("template render error: {0}")]
    Template(String),

    #[error("template parse error: {0}")]
    TemplateSyntax(String),

    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("utf-8 decoding error at byte {valid_up_to}")]
    Utf8 { valid_up_to: usize },

    #[error("{0}")]
    Other(String),
}

impl TokenizerError {
    pub fn backend<E: std::fmt::Display>(e: E) -> Self {
        Self::Backend(e.to_string())
    }

    pub fn template<E: std::fmt::Display>(e: E) -> Self {
        Self::Template(e.to_string())
    }

    pub fn template_syntax<E: std::fmt::Display>(e: E) -> Self {
        Self::TemplateSyntax(e.to_string())
    }

    pub fn invalid_arg<E: std::fmt::Display>(e: E) -> Self {
        Self::InvalidArgument(e.to_string())
    }

    pub fn io_at(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Self::Io { path: path.into(), source }
    }
}

impl From<minijinja::Error> for TokenizerError {
    fn from(e: minijinja::Error) -> Self {
        TokenizerError::Template(format!("{e:#}"))
    }
}

impl From<tokenizers::Error> for TokenizerError {
    fn from(e: tokenizers::Error) -> Self {
        TokenizerError::Backend(e.to_string())
    }
}

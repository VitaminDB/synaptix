use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum GgufError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("{path}: не GGUF-файл (magic {got:02x?})")]
    BadMagic { path: PathBuf, got: [u8; 4] },

    #[error("GGUF версии {0} не поддерживается (нужна 2 или 3)")]
    BadVersion(u32),

    #[error("файл обрезан: нужно {need} байт с оффсета {at}, доступно {have}")]
    Truncated { at: usize, need: usize, have: usize },

    #[error("неизвестный тип GGUF-значения: {0}")]
    BadValueType(u32),

    #[error("неизвестный ggml-тип тензора: {0}")]
    BadTensorType(u32),

    #[error("ggml-тип {0} пока не поддержан деквантизацией (нужны таблицы-решётки IQ)")]
    UnsupportedQuant(&'static str),

    #[error("метаданные: ключ `{0}` отсутствует")]
    MissingKey(String),

    #[error("метаданные: ключ `{key}` имеет тип {actual}, ожидался {expected}")]
    WrongKeyType {
        key: String,
        expected: &'static str,
        actual: &'static str,
    },

    #[error("тензор `{name}`: {reason}")]
    BadTensor { name: String, reason: String },

    #[error("архитектура `{0}` не поддержана конвертером")]
    UnsupportedArch(String),

    #[error("utf-8 в GGUF-строке: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),

    #[error("bundle: {0}")]
    Bundle(String),

    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<synaptix_bundle::Error> for GgufError {
    fn from(e: synaptix_bundle::Error) -> Self {
        Self::Bundle(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, GgufError>;

use std::path::PathBuf;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("not a .syn bundle (bad magic at {where_:?})")]
    BadMagic { where_: &'static str },

    #[error("unsupported format version {found_major}.{found_minor} (max supported {supported_major})")]
    UnsupportedFormatVersion {
        found_major: u16,
        found_minor: u16,
        supported_major: u16,
    },

    #[error("unsupported required capability `{0}`")]
    UnsupportedCapability(String),

    #[error("file too small ({0} bytes) to be a .syn bundle")]
    FileTooSmall(u64),

    #[error("central directory crc mismatch (expected {expected:#x}, got {got:#x})")]
    CdirCrcMismatch { expected: u32, got: u32 },

    #[error("footer self-check crc mismatch (expected {expected:#x}, got {got:#x})")]
    FooterCrcMismatch { expected: u32, got: u32 },

    #[error("chunk #{id} payload crc mismatch (expected {expected:#x}, got {got:#x})")]
    ChunkCrcMismatch { id: u64, expected: u32, got: u32 },

    #[error("chunk #{id} bad magic (expected SYNK, got {got:?})")]
    ChunkBadMagic { id: u64, got: [u8; 4] },

    #[error("cdir offset {0} out of bounds")]
    CdirOutOfBounds(u64),

    #[error("cbor decode: {0}")]
    CborDecode(String),

    #[error("cbor encode: {0}")]
    CborEncode(String),

    #[error("invalid path `{path}`: {reason}")]
    InvalidPath { path: String, reason: &'static str },

    #[error("file `{0}` not found in bundle")]
    FileNotFound(String),

    #[error("tensors chunk not found in bundle")]
    TensorsChunkMissing,

    #[error("safetensors: {0}")]
    Safetensors(String),

    #[error("path `{0}` is not utf-8")]
    NonUtf8Path(PathBuf),

    #[error("platform unsupported: {0}")]
    PlatformUnsupported(&'static str),

    #[error("bundle is locked by another editor: {0}")]
    BundleBusy(PathBuf),
}

impl From<safetensors::SafeTensorError> for Error {
    fn from(value: safetensors::SafeTensorError) -> Self {
        Error::Safetensors(value.to_string())
    }
}

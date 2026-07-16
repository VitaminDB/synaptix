//! `.syn` — single-file model bundle format.
//!
//! Layout: fixed `FileHeader` (64B) → variable chunks → CBOR central directory
//! → fixed `Footer` (40B) at EOF.
//!
//! Reader path is zero-copy: a single `mmap` over the whole file. Callers
//! get byte slices for each component and feed them into their own tensor
//! backend (synaptix-core / safetensors).
//!
//! Editing is O(cdir_size) — tombstone the entry, rewrite cdir + footer.
//! The tensor payload is never touched on a `remove_file`.

pub mod cdir;
pub mod chunk;
pub mod error;
pub mod header;
pub mod path;
pub mod quantized;

mod bundle;
mod builder;
mod editor;
mod resolver;


pub use bundle::{Bundle, DirEntry};
pub use builder::{
    resolve_safetensors_in_dir, BundleBuilder, ProgressCallback, ProgressEvent,
};

pub fn available_space(path: impl AsRef<std::path::Path>) -> Result<u64> {
    fs4::available_space(path).map_err(Error::Io)
}
pub use cdir::{BundleMeta, CdirFormat, ChunkEntry, ChunkStatus, ChunkType, FileTag, LoraOverlay, RefSpec};
pub use editor::{compact, BundleEditor};
pub use error::{Error, Result};
pub use quantized::{QuantizedChunkHeader, QUANT_FORMAT_FP8E4M3, QUANTIZED_CHUNK_NAME};
pub use resolver::{FsResolver, RefResolver};

pub const SUPPORTED_MAJOR: u16 = 1;
pub const CURRENT_MINOR: u16 = 0;

pub const SUPPORTED_REQUIRED_CAPS: &[&str] = &[CAP_TENSOR_DELTA];

pub const CAP_SHA256_MANIFEST: &str = "sha256-manifest";
pub const CAP_BLAKE3_MANIFEST: &str = "blake3-chunks";
pub const CAP_TENSOR_DELTA: &str = "tensor-delta-overlays";

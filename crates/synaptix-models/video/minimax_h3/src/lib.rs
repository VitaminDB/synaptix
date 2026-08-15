pub mod adaln;
pub mod audio_vae;
pub mod config;
pub mod dit;
pub mod guider;
pub mod layout;
pub mod loader;
pub mod memory;
pub mod pipeline;
pub mod rope;
pub mod runtime;
pub mod scheduler;
pub mod source;
pub mod spec;
pub mod text_encoder;
pub mod vae;

pub use config::{AudioVaeConfig, H3Config, H3Variant, VaeConfig};
pub use layout::{PackedLayout, SegmentKind};
pub use loader::{H3Checkpoint, H3Paths, LoraWeights};
pub use source::{H3Component, H3EncoderSource, H3Source};
pub use memory::H3MemoryMode;
pub use scheduler::{time_shift_sigma, H3Scheduler};

pub type Result<T> = std::result::Result<T, H3Error>;

#[derive(Debug, thiserror::Error)]
pub enum H3Error {
    #[error("io: {0}")]
    Io(String),
    #[error("load: {0}")]
    Load(String),
    #[error("config: {0}")]
    Config(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("layout: {0}")]
    Layout(String),
    #[error("отменено")]
    Cancelled,
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
}

pub mod config;
pub mod conv;
pub mod generate;
pub mod graph;
pub mod head;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod processor;
pub mod qwen2;
pub mod schedule;
pub mod vae;

pub use config::{GenerationConfig, PreprocessorConfig, VibeVoiceConfig};
pub use generate::{GenerationOutput, SpeechGenerator};
pub use loader::VibeVoiceCheckpoint;
pub use model::VibeVoiceModel;
pub use pipeline::{VibeVoicePipeline, VoiceSample};
pub use processor::{parse_script, plain_text_to_script, ScriptLine, VibeVoiceProcessor};

pub type Result<T> = std::result::Result<T, VibeVoiceError>;

#[derive(Debug, thiserror::Error)]
pub enum VibeVoiceError {
    #[error("config: {0}")]
    Config(String),
    #[error("bundle: {0}")]
    Bundle(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("audio: {0}")]
    Audio(String),
    #[error("inference: {0}")]
    Inference(String),
}

pub(crate) fn err<E: std::fmt::Display>(e: E) -> VibeVoiceError {
    VibeVoiceError::Inference(e.to_string())
}

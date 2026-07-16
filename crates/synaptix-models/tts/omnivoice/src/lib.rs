//! OmniVoice — massively multilingual zero-shot TTS (k2-fsa) на примитивах synaptix.
//!
//! Masked-diffusion language model: двунаправленный Qwen3-бэкбон над 8 RVQ-аудио-кодбуками
//! + итеративное параллельное раскрытие маскированных токенов (CFG, num_step шагов) +
//! нейро-кодек HiggsAudioV2 (encode ref-аудио → коды, decode коды → волна 24 кГц).
//! Режимы: voice-cloning / voice-design / auto.
//!
//! Источник истины — официальный upstream (k2-fsa OmniVoice + HF HiggsAudioV2TokenizerModel),
//! по upstream-спеке. Полная архитектура, раскладка весов и план порта — в `SPEC.md`.
//!
//! Статус: скелет + config. Реализация компонентов — поэтапно (см. SPEC.md «План порта»).

pub mod audio_codec;
pub mod audio_encode;
pub mod backbone;
pub mod config;
pub mod loader;
pub mod masked_decode;
pub mod pipeline;
pub mod prompt;
pub mod text;

pub use audio_codec::CodecDecoder;
pub use audio_encode::CodecEncoder;
pub use config::{HiggsAudioConfig, OmniVoiceConfig, OmniVoiceGenerationConfig};
pub use pipeline::OmniVoicePipeline;
pub use prompt::{add_punctuation, create_voice_clone_prompt, remove_silence, VoiceClonePrompt};
pub use text::{combine_text, DurationEstimator, PreparedInputs, TextFrontend};

pub type Result<T> = std::result::Result<T, OmniVoiceError>;

#[derive(Debug, thiserror::Error)]
pub enum OmniVoiceError {
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
    #[error("other: {0}")]
    Other(String),
}

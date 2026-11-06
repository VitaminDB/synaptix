//! Audio8-TTS (HF `Edge0/Audio8-TTS-Realistic-Air`, `model_type = arktts`) — нативный порт.
//!
//! DualAR: медленный трансформер (Qwen2.5-0.5B-подобный, 24 слоя) предсказывает на кадр
//! один семантический токен, быстрый (4 слоя) — девять акустических кодбуков кадра;
//! кодек 44.1 кГц (2048 семплов на кадр, ~21.5 кадра/с) декодирует коды в звук и кодирует
//! образец голоса. Спецификация и сверка — `SPEC.md`.

pub mod codec;
pub mod config;
pub mod dualar;
pub mod loader;
pub mod pipeline;
pub mod sample;
pub mod text;

pub use config::Audio8Config;
pub use pipeline::{Audio8Pipeline, GenerateOptions, VoicePrompt};

use synaptix_core::error::SynaptixError;

#[derive(Debug, thiserror::Error)]
pub enum Audio8Error {
    #[error("config: {0}")]
    Config(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tensor: {0}")]
    Tensor(#[from] SynaptixError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("audio: {0}")]
    Audio(String),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, Audio8Error>;

/// Частота звука кодека.
pub const SAMPLE_RATE: u32 = 44_100;
/// Семплов на кадр модели.
pub const FRAME_SAMPLES: usize = 2048;

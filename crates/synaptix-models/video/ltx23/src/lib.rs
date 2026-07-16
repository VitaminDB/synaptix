//! LTX-2.3 22B — совместная генерация видео+звук (text/image → video + audio),
//! нативная реализация на synaptix.
//!
//! Чекпойнт `ltx-2.3-22b-distilled` самодостаточен и под одним файлом держит все
//! подмодели (префиксы `model.diffusion_model` — DiT `AVTransformer3DModel`,
//! `vae` — `CausalVideoAutoencoder`, `audio_vae`, `vocoder`,
//! `text_embedding_projection`). Текст-энкодер — внешняя Gemma-3-12B. Конфиг
//! читается из `__metadata__["config"]` весов (см. [`config`]); веса грузятся
//! zero-copy через [`loader::LtxCheckpoint`].
//!
//! Порт ведётся видео-частью вперёд (Фазы 1-7), звук — позже (Фазы 8-11).

pub mod audio_vae;
pub mod config;
pub mod dit;
pub mod guider;
pub mod loader;
pub mod vae;
pub mod vocoder;
pub mod model;
pub mod pipeline;
pub mod runtime;
pub mod spec;
pub mod text_encoder;
pub mod upscaler;

pub use config::Ltx23Config;
pub use loader::LtxCheckpoint;

pub type Result<T> = std::result::Result<T, LtxError>;

#[derive(Debug, thiserror::Error)]
pub enum LtxError {
    #[error("io: {0}")]
    Io(String),
    #[error("load: {0}")]
    Load(String),
    #[error("config: {0}")]
    Config(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("отменено")]
    Cancelled,
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
}

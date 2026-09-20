//! YuE2 — генерация песен из стиля и лирики через редактируемую партитуру.
//!
//! Один костяк AR–NAR Mixture-of-Transformers делает две разные работы: AR-путь
//! авторегрессивно пишет партитуру (ABC) и семантические токены музыки, NAR-путь
//! тем же телом, но своим комплектом весов, строит акустические латенты методом
//! flow matching. Латенты превращает в 48 кГц стерео Oobleck-VAE.
//!
//! Раскладка по модулям:
//!
//! - [`protocol`] — грамматика промпта, спец-токены, дефолты сэмплинга;
//! - [`tokenizer`] — текстовый BPE (`qwen.tiktoken`);
//! - [`ar`] — авторегрессивные фазы поверх общего `DecoderModel` движка;
//! - [`nar`] — flow matching: AR-контекст в KV-кэше, midpoint-решатель;
//! - [`vae`] — Oobleck-декодер с тайлингом (и энкодер — для каверов);
//! - [`pipeline`] — стадии целиком: план → музыка → латенты → звук.

pub mod ar;
pub mod config;
pub mod loader;
pub mod nar;
pub mod pipeline;
pub mod protocol;
pub mod tokenizer;
pub mod vae;

use synaptix_core::error::SynaptixError;

#[derive(Debug, thiserror::Error)]
pub enum YueError {
    #[error("конфигурация: {0}")]
    Config(String),
    #[error("загрузка: {0}")]
    Load(String),
    #[error("тензор: {0}")]
    Tensor(#[from] SynaptixError),
    #[error("прервано: {0}")]
    Cancelled(&'static str),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, YueError>;

pub use config::{Yue2Config, Yue2VaeConfig};
pub use pipeline::{Callbacks, SongResult, Yue2Options, Yue2Paths, Yue2Pipeline};
pub use protocol::{Cot, GenerationConfig, Sampling, SongRequest};
pub use tokenizer::Yue2Tokenizer;
pub use vae::Yue2Vae;

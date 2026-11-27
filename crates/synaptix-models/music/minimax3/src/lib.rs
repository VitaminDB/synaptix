//! MiniMax Music 3 — песни по тексту и описанию (до ~6 минут, 44,1 кГц стерео).
//!
//! Путь (эталон — модульный пайплайн diffusers, PR #14456):
//! промпт → глобальная LM (Qwen3 8B, первый кодбук RVQ покадрово, CFG) и
//! локальная LM (глубинный декодер, кодбуки 2…8) → скрытые состояния кадра
//! (8 × 4096) → кодер условий → flow-matching трансформер по окнам с
//! перекрытием → DAC-декодер (Flow-VAE) → волна.
//!
//! - [`prompt`] — сборка промпта и токены чекпойнта;
//! - [`ar`] — авторегрессия: глобальная LM и глубинный декодер;
//! - [`flow`] — кодер условий, трансформер, сэмплер по окнам;
//! - [`vocoder`] — DAC-декодер;
//! - [`pipeline`] — всё вместе.

pub mod ar;
pub mod flow;
pub mod loader;
pub mod pipeline;
pub mod prompt;
pub mod vocoder;

pub use pipeline::{Music3, Music3Request, Music3Output};

#[derive(Debug, thiserror::Error)]
pub enum Music3Error {
    #[error("загрузка: {0}")]
    Load(String),
    #[error("запрос: {0}")]
    Input(String),
    #[error("остановлено")]
    Cancelled,
    #[error("{0}")]
    Other(String),
    #[error(transparent)]
    Core(#[from] synaptix_core::error::SynaptixError),
}

pub type R<T> = Result<T, Music3Error>;

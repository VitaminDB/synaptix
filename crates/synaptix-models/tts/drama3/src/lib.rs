//! Fish Audio Drama 3 (`fishaudio/drama-3`, `FishOmniModel`) — речь по тексту
//! с клонированием голоса по образцу, тегами говорящих `<|speaker:N|>` и
//! событий (`[laughs]`), 44,1 кГц моно.
//!
//! Путь (эталон — sglang-omni `models/fishaudio_drama3`): образец → кодек →
//! коды `[10, T]`; промпт строками-кадрами `[токен, c0..c9]`; основа
//! Qwen3.5-MoE (40 слоёв, 2048) на каждой позиции даёт состояние после
//! финальной нормы; голова по 4097 кандидатам (eos и полоса семантики) —
//! токен кадра; башня `codebook_mtp_shared` (4 общих слоя + 9 по кодбукам,
//! каузальные по времени) — коды 1…9; кодек — волна.
//!
//! - [`prompt`] — строки промпта;
//! - [`tower`] — башня кодбуков;
//! - [`text`] — разбиение длинного текста на куски;
//! - [`sample`] — top-k/top-p и RAS;
//! - [`pipeline`] — всё вместе.

pub mod pipeline;
pub mod prompt;
pub mod sample;
pub mod text;
pub mod tower;

pub use pipeline::{Drama3, Drama3Output, Drama3Request, Reference};

/// Частота звука кодека.
pub const SAMPLE_RATE: u32 = 44_100;
/// Семплов на кадр.
pub const FRAME_SAMPLES: usize = 2048;
pub const CODEBOOKS: usize = 10;
pub const CODEBOOK_SIZE: usize = 4096;
/// Размер остаточных кодбуков кодека (головы шире — коды клампятся).
pub const RESIDUAL_SIZE: u32 = 1024;
pub const SEMANTIC_BEGIN: u32 = 248_099;
pub const SEMANTIC_END: u32 = 252_194;
pub const EOS: u32 = 248_046;

/// Строка промпта/кадр: `[токен, c0..c9]`.
pub type Row = [u32; CODEBOOKS + 1];

pub fn is_semantic(token: u32) -> bool {
    (SEMANTIC_BEGIN..=SEMANTIC_END).contains(&token)
}

#[derive(Debug, thiserror::Error)]
pub enum Drama3Error {
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

pub type R<T> = Result<T, Drama3Error>;

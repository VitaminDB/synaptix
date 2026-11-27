//! Seed-VC v2 — преобразование голоса по образцу без дообучения (22,05 кГц моно).
//!
//! Путь (эталон — `Plachtaa/seed-vc`, `modules/v2/vc_wrapper.py`):
//! речь 16 кГц → HuBERT-large (18 слоёв) → ASTRAL (ConvNeXt + BSQ): «широкие»
//! токены содержания (2048 кодов) и «узкие» (32 кода) → [при `convert_style`
//! авторегрессия переписывает широкие токены источника в манере образца] →
//! регулятор длины → flow-matching DiT (условия: токены, мел образца, вектор
//! диктора CAM++; CFG по содержанию и по голосу) → BigVGAN → волна.
//!
//! - [`mel`] — мел-спектрограмма 22,05 кГц;
//! - [`campplus`] — вектор диктора (kaldi fbank → CAM++);
//! - [`astral`] — HuBERT + токенизаторы содержания;
//! - [`regulator`] — регуляторы длины;
//! - [`ar`] — авторегрессия манеры;
//! - [`cfm`] — DiT и сэмплер;
//! - [`vocoder`] — BigVGAN;
//! - [`pipeline`] — всё вместе;
//! - [`v1`] — Seed-VC v1 (whisper-small + DiT с WaveNet-постнетом): только тембр.

pub mod ar;
pub mod astral;
pub mod campplus;
pub mod cfm;
pub mod loader;
pub mod mel;
pub mod nn;
pub mod pipeline;
pub mod regulator;
pub mod v1;
pub mod vocoder;

pub use pipeline::{SeedVc, SeedVcOutput, SeedVcRequest};
pub use v1::{SeedVcV1, SeedVcV1Request};

#[derive(Debug, thiserror::Error)]
pub enum SeedVcError {
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

pub type R<T> = Result<T, SeedVcError>;

/// Частота выхода и мел-спектрограммы, Гц.
pub const SAMPLE_RATE: u32 = 22_050;
/// Шаг мел-кадра в сэмплах.
pub const HOP: usize = 256;
/// Число мел-полос.
pub const N_MELS: usize = 80;

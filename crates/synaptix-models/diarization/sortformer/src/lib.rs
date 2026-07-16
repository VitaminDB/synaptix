//! NVIDIA NeMo **Streaming Sortformer** speaker diarization (4spk v2.1) — нативная
//! реализация на примитивах synaptix.
//!
//! Цепочка: PCM 16kHz → NeMo mel (128-bin Slaney) → FastConformer-энкодер (8× dw-striding
//! subsampling + 17 Conformer-слоёв с T5-XL rel-pos attention) → Sortformer-head (18 post-LN
//! transformer-слоёв + classifier) → sigmoid → per-speaker probs @ 12.5 Hz → сегменты.
//!
//! Веса из `.syn`-бандла в HF-раскладке (`encoder.layers.{i}.*`, `head.layers.{i}.*`).
//! Источник истины — официальный NVIDIA NeMo. Маппинг HF→NeMo-имён и
//! детали архитектуры — в `SPEC.md`; bit-exact гейт — `tests/sortformer_gate.rs`.
//!
//! Статус: BATCH/full-attention путь реализован и сверен с NeMo (bin-agree спикеров = 1.0).
//! Streaming-режим (для длинных записей) — `streaming.rs`, фаза 2.

pub mod config;
pub mod encoder;
pub mod head;
pub mod loader;
pub mod mel;
pub mod model;
pub mod pipeline;
pub mod postprocess;
pub mod streaming;

pub use config::SortformerConfig;
pub use loader::SortformerWeights;
pub use model::SortformerModel;
pub use pipeline::SortformerPipeline;
pub use postprocess::{DiarizationResult, DiarizeSegment, PostprocessParams};

pub type Result<T> = std::result::Result<T, SortformerError>;

#[derive(Debug, thiserror::Error)]
pub enum SortformerError {
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
    #[error("other: {0}")]
    Other(String),
}

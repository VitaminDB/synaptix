//! FLUX.1 (dev / schnell) — text→image, нативная реализация на synaptix.
//!
//! Тяжёлые нейрокомпоненты (CLIP-L pooled, T5-XXL encoder, MMDiT
//! FluxTransformer2DModel, AutoencoderKL-16ch) живут в [`synaptix_nn`] и
//! проверяются bit-exact к HF diffusers/transformers НА CUDA (см.
//! `tests/ref_*.rs`). Здесь — пайплайн поверх них: токенайзеры (CLIP BPE +
//! T5 sentencepiece), источник весов — каталог diffusers или `.syn`-бандл
//! ([`source`]/[`loader`]), стадии для нод ([`model`]), flow-matching
//! планировщик и txt2img одним вызовом ([`pipeline`]).
//!
//! FLUX.1-dev — guidance-distilled: БЕЗ CFG/negative, один forward на шаг,
//! guidance подаётся как эмбеддинг. FLUX.1-schnell — без guidance-эмбеддера,
//! статический сдвиг расписания; различаются по конфигам модели.

pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod scheduler;
pub mod source;
pub mod t5;
pub mod tokenizer;
pub mod transformer;

pub use config::Txt2ImgParams;
pub use model::{FluxConditioning, FluxModel, OffloadMode, SampleParams};
pub use pipeline::FluxPipeline;
pub use source::{FluxComponent, FluxSource};
pub use transformer::FluxTransformer;
pub use tokenizer::ClipTokenizer;

pub type Result<T> = std::result::Result<T, FluxError>;

#[derive(Debug, thiserror::Error)]
pub enum FluxError {
    #[error("io: {0}")]
    Io(String),
    #[error("load: {0}")]
    Load(String),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("config: {0}")]
    Config(String),
    #[error("tensor: {0}")]
    Tensor(#[from] synaptix_core::error::SynaptixError),
    #[error("diffusion: {0}")]
    Diffusion(#[from] synaptix_diffusion::DiffusionError),
    #[error("отменено")]
    Cancelled,
}

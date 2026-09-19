//! SDXL — Stable Diffusion XL (text→image), нативная реализация на synaptix.
//!
//! Тяжёлые нейрокомпоненты (CLIP-L + bigG text-энкодеры, UNet2DConditionModel,
//! AutoencoderKL) живут в [`synaptix_nn`] и проверены bit-exact к HF
//! diffusers/transformers (см. `tests/ref_{clip,vae,unet}.rs`). Здесь собирается
//! пайплайн поверх них: CLIP-токенайзер ([`tokenizer`]), источник весов —
//! HF-каталог или `.syn`-бандл ([`source`]), Euler SDXL ([`scheduler`]),
//! txt2img одним вызовом ([`model`]/[`pipeline`], CLI `synaptix imagine`) и
//! стадии для нод с img2img ([`stages`]).

pub mod config;
pub mod model;
pub mod pipeline;
pub mod scheduler;
pub mod source;
pub mod stages;
pub mod tokenizer;

pub use config::Txt2ImgParams;
pub use model::SdxlModel;
pub use pipeline::SdxlPipeline;
pub use source::SdxlSource;
pub use stages::{SdxlCheckpoint, SdxlConditioning, SdxlSampleParams, SdxlUnet};
pub use tokenizer::ClipTokenizer;

pub type Result<T> = std::result::Result<T, SdxlError>;

#[derive(Debug, thiserror::Error)]
pub enum SdxlError {
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

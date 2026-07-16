//! SDXL — Stable Diffusion XL (text→image), нативная реализация на synaptix.
//!
//! Тяжёлые нейрокомпоненты (CLIP-L + bigG text-энкодеры, UNet2DConditionModel,
//! AutoencoderKL) живут в [`synaptix_nn`] и проверены bit-exact к HF
//! diffusers/transformers (см. `tests/ref_{clip,vae,unet}.rs`). Здесь собирается
//! txt2img-пайплайн поверх них: CLIP-токенайзер ([`tokenizer`]), загрузка весов
//! из HF-директории ([`loader`]/[`model`]), CFG + Euler-планировщик и VAE-декод
//! ([`pipeline`]).

pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;
pub mod tokenizer;

pub use config::Txt2ImgParams;
pub use model::SdxlModel;
pub use pipeline::SdxlPipeline;
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
}

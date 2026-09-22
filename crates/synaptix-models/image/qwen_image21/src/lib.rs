//! Qwen-Image 2.1 (Alibaba Qwen, 2026): одна модель на картинку по тексту,
//! правку по референсам (до 10 картинок) и прозрачные RGBA-картинки.
//! Нативно на synaptix.
//!
//! Устройство (по diffusers `QwenImage21Pipeline`):
//! - энкодер — Qwen3-VL 8B: картинки-референсы проходят башню зрения
//!   (SigLIP-подобный ViT, 27 блоков, deepstack в LLM) и вставляются в
//!   промпт на место `<|image_pad|>`; кондиционирование — выход последнего
//!   слоя LLM **до** финальной нормы, без системной части шаблона
//!   ([`text_encoder`], [`vision`]);
//! - DiT — 32 single-stream блока (7 млрд параметров), одна общая
//!   модуляция на все блоки, блочно-причинное внимание: текст причинный,
//!   каждая картинка (референс и генерируемая) внутри себя двунаправленная;
//!   токены референсов и текста модулируются временем 0, поэтому их K/V
//!   считаются один раз и кэшируются на все шаги ([`transformer`]);
//! - VAE — резидуальный 3D-VAE (Wan 2.2) в режиме одного кадра: 4 канала
//!   RGBA, сжатие 16×, латент на 64 канала без упаковки 2×2 ([`vae`]).
//!
//! Память: блоки DiT, не влезшие в VRAM, стримятся (пиннованная копия на
//! хосте или чтение из mmap-источника), слои энкодера — из источника. См.
//! [`model::QwenImage21Model`].

pub mod config;
pub mod image;
pub mod model;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;

pub use config::{Qwen21Config, Qwen21VaeConfig, Qwen3VlConfig, Qwen3VlVisionConfig};
pub use image::RgbaImage;
pub use model::{Qwen21Conditioning, Qwen21References, QwenImage21Model, SampleParams};
pub use synaptix_image_qwen::transformer::Placement as MemoryMode;
pub use synaptix_image_qwen::QwenImageSource;
pub use transformer::QwenImage21Transformer;

pub type Result<T> = std::result::Result<T, QwenImage21Error>;

#[derive(Debug, thiserror::Error)]
pub enum QwenImage21Error {
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

impl From<synaptix_image_qwen::QwenImageError> for QwenImage21Error {
    fn from(e: synaptix_image_qwen::QwenImageError) -> Self {
        use synaptix_image_qwen::QwenImageError as E;
        match e {
            E::Load(s) => Self::Load(s),
            E::Tokenizer(s) => Self::Tokenizer(s),
            E::Config(s) => Self::Config(s),
            E::Tensor(t) => Self::Tensor(t),
            E::Diffusion(d) => Self::Diffusion(d),
            E::Cancelled => Self::Cancelled,
        }
    }
}

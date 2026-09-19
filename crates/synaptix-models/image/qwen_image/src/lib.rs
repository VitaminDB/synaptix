//! Qwen-Image (Alibaba Qwen): правка картинки по инструкции — Qwen-Image-Edit
//! (08.2025) и Qwen-Image-Edit-2509/2511 (несколько картинок), плюс картинка
//! по тексту у базового Qwen-Image. Нативно на synaptix.
//!
//! Устройство:
//! - энкодер — Qwen2.5-VL 7B: картинки проходят башню зрения и вставляются в
//!   промпт на место `<|image_pad|>`, кондиционирование — последнее скрытое
//!   состояние LLM без системной части шаблона ([`text_encoder`], [`vision`]);
//! - DiT — MMDiT на 60 double-stream блоков (20 млрд параметров), картинки
//!   для правки идут в ту же последовательность токенов, что и генерируемая,
//!   со своим индексом кадра в RoPE ([`transformer`]);
//! - VAE — каузальный 3D-VAE Wan 2.1 на 16 каналов в режиме одного кадра
//!   ([`vae`]); латент пакуется 2×2 в 64 канала.
//!
//! Память: модуляции блоков считаются один раз на расписание, остальные веса
//! блоков DiT, не влезшие в VRAM, стримятся (пиннованная копия на хосте или
//! чтение из mmap-источника), слои энкодера — из источника. См.
//! [`model::QwenImageModel`].

pub mod config;
pub mod memory;
pub mod model;
pub mod preprocess;
pub mod scheduler;
pub mod source;
pub mod text_encoder;
pub mod transformer;
pub mod vae;
pub mod vision;

pub use config::{QwenImageConfig, QwenImageVariant, TextEncoderConfig};
pub use model::{MemoryMode, QwenConditioning, QwenImageModel, QwenReferences, SampleParams};
pub use source::QwenImageSource;
pub use transformer::QwenImageTransformer;

pub type Result<T> = std::result::Result<T, QwenImageError>;

#[derive(Debug, thiserror::Error)]
pub enum QwenImageError {
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

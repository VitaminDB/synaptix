//! FLUX.2 (Black Forest Labs) — dev и klein 4B/9B: картинка по тексту и
//! правка по референсным картинкам, нативно на synaptix.
//!
//! Отличия от FLUX.1 ([`synaptix-image-flux`]), из-за которых это отдельный
//! крейт, а не вариант:
//! - текст кодирует LLM: Mistral-Small-3.1 24B у dev, Qwen3 4B/8B у klein;
//!   кондиционирование — склейка скрытых состояний трёх слоёв
//!   (dev 10/20/30, klein 9/18/27), CLIP/pooled нет;
//! - DiT: общая на все блоки модуляция (по одной на double img/txt и single),
//!   линейные слои без bias, SwiGLU вместо GELU, RoPE по четырём осям
//!   (t, h, w, l) с theta 2000;
//! - VAE на 32 канала; латент пакуется 2×2 в 128 каналов и нормируется
//!   статистикой BatchNorm VAE;
//! - референсные картинки идут в ту же последовательность токенов с
//!   координатой `t = 10, 20, …` — так работает правка.
//!
//! Память: блоки DiT, не влезшие в VRAM, стримятся (пиннованная копия на
//! хосте или чтение из mmap-источника), слои текстового энкодера — из
//! mmap-источника по одному; тем самым вся линейка работает на картах от
//! ~7 ГБ. См. [`model::Flux2Model`].

pub mod config;
pub mod memory;
pub mod model;
pub mod scheduler;
pub mod source;
pub mod text_encoder;
pub mod transformer;
pub mod vae;

pub use config::{Flux2Config, Flux2Variant, TextEncoderConfig};
pub use model::{Flux2Conditioning, Flux2Model, Flux2References, MemoryMode, SampleParams};
pub use source::Flux2Source;
pub use transformer::Flux2Transformer;

pub type Result<T> = std::result::Result<T, Flux2Error>;

#[derive(Debug, thiserror::Error)]
pub enum Flux2Error {
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

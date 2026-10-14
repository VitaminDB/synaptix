//! Башня зрения MiniCPM-V 4.7 (OpenBMB): SigLIP-энкодер NaViT с оконным
//! объединителем после 7-го слоя, объединитель 2×2 в пространство LLM,
//! препроцессор с нарезкой картинки на миниатюру и срезы, строка-заполнитель
//! промпта и «холстовые» позиции M-RoPE. Порт `modeling_minicpmv4_6/4_7.py`,
//! `image_processing_minicpmv4_6.py`, `processing_minicpmv4_6.py` из HF
//! transformers.

pub mod config;
pub mod layout;
pub mod loader;
pub mod model;
pub mod preprocess;

pub use config::{ConfigError, MiniCpmConfig};
pub use layout::{
    canvas_positions, placeholder, placeholder_token_count, CanvasIds, CanvasPositions,
    PlaceholderTokens,
};
pub use loader::{bundle_has_vision, load_from_bundle, BundleVisionWeights};
pub use model::{MiniCpmVision, VisionError, VisionWeights};
pub use preprocess::{
    prepare_image, prepare_tensor, CropGrid, ImageLayout, PreparedImage, PreprocessError,
    PreprocessLimits,
};

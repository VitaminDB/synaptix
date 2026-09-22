//! Типы ggml живут в `synaptix_core::quant::ggml` (нужны движку для
//! исполнения GGUF напрямую); здесь — реэкспорт для прежних путей.

pub use synaptix_core::quant::ggml::{GgmlType, K_SCALE_SIZE, QK_K};

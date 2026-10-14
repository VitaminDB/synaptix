pub mod config;
pub mod loader;
pub mod model;
pub use synaptix_llm_common::mrope;
pub mod pipeline;

/// Башня зрения MiniCPM-V 4.7 — реэкспорт для вызывающих (раскладка
/// картинки, заполнитель), чтобы не тянуть крейт башни напрямую.
pub use synaptix_vlm_minicpm as vlm_minicpm;

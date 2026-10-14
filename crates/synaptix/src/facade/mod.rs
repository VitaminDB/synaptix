//! Высокоуровневые фасады поверх нативных synaptix-моделей. Каждая подсистема
//! (LLM / ASR / embeddings / rerank / диаризация / TTS) даёт стабильный
//! публичный API + детекцию архитектуры, диспетчащую в правильный нативный
//! pipeline. Добавление новой модели = новая ветка `match` ВНУТРИ synaptix;
//! потребитель (synthos, CLI) кода не меняет.

pub mod arch;
pub mod asr;
pub mod catalog;
pub mod chat;
#[cfg(feature = "engine")]
pub mod device;
#[cfg(feature = "engine")]
pub mod diarization;
pub mod embedding;
pub mod llm;
pub mod pack;
pub mod rerank;
pub mod params;
pub mod sampling;
#[cfg(feature = "engine")]
pub mod tts;

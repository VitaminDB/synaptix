//! Высокоуровневые фасады поверх нативных synaptix-моделей. Каждая подсистема
//! (LLM / ASR / embeddings / rerank / диаризация / TTS) даёт стабильный
//! публичный API + детекцию архитектуры, диспетчащую в правильный нативный
//! pipeline. Добавление новой модели = новая ветка `match` ВНУТРИ synaptix;
//! потребитель (synthos, CLI) кода не меняет.

pub mod arch;
pub mod asr;
pub mod diarization;
pub mod embedding;
pub mod llm;
pub mod rerank;
pub mod tts;

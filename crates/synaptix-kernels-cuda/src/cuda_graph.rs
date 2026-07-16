//! CUDA Graph capture + replay — зарезервированный модуль.
//!
//! В synaptix inline-реализации графа пока нет (валидированный graph-decode для
//! Qwen3.6 живёт в крейте `ai-quant`/`llm-qwen36`, не здесь), поэтому выносить
//! нечего. Модуль оставлен под будущий device-resident capture/replay поверх
//! `crate::stream_pool`. См. план SSM/linear-attn (follow-up).

//! Stream lifecycle + event sync — зарезервированный модуль.
//!
//! Inline-реализации пула стримов в synaptix пока нет — ядра берут стрим из
//! `synaptix_core::device::cuda::default_stream`. Модуль оставлен под будущий
//! multi-stream pool (overlap H2D/compute) и event-синхронизацию. Выносить
//! нечего (нет исходного inline-кода).

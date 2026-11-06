//! LiquidAI LFM3: гибрид коротких свёрток (LFM2-ShortConv, k = 7) и внимания (full с частичным RoPE
//! «нулевой хвост» и sliding 256, Q/K-норма, sigmoid-гейт на голову), сэндвич-нормы, связанная голова.
//! Работает на общем [`synaptix_llm_common::DecoderModel`]: веса LFM3 переименовываются при загрузке.

pub mod config;
pub mod loader;
pub mod model;
pub mod pipeline;

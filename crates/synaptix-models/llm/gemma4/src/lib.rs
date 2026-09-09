//! Gemma-4 — семейство Google DeepMind (2026): MoE-ветка рядом с плотным MLP,
//! чередование sliding/global слоёв с РАЗНОЙ геометрией голов, proportional
//! RoPE и общий K/V на global-слоях.

pub mod config;
pub mod loader;
pub mod pipeline;

pub use config::{ConfigError, Gemma4Config, LayerType};
pub use loader::{is_bundle, read_aux, Gemma4Weights, LoadError};
pub use pipeline::{Gemma4Pipeline, PipelineError};

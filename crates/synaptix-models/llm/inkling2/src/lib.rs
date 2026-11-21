//! Inkling-2 (thinkingmachines, `inkling2_loop`): зацикленный MoE-трансформер.
//! Схема и проверка паритета — `docs/INKLING2-MINI-PORT.md`.

pub mod config;
pub mod kern;
pub mod loader;
pub mod model;

pub use config::Inkling2Config;
pub use loader::{Inkling2Weights, LoadError};
pub use model::{Inkling2Model, ModelCache};

//! Inkling-2 (thinkingmachines, `inkling2_loop`): зацикленный MoE-трансформер.
//! Схема и проверка паритета — `docs/INKLING2-MINI-PORT.md`.

pub mod config;
pub mod kern;
pub mod loader;
pub mod model;
pub mod pack;
pub mod pipeline;
pub mod preprocess;
pub mod vision;

pub use config::Inkling2Config;
pub use loader::{Inkling2Weights, LoadError};
pub use model::{Inkling2Model, ModelCache};
pub use pipeline::{Inkling2Pipeline, Inkling2Session, PipelineError};
pub use preprocess::{ImageLimits, PreparedImage};
pub use vision::VisionTower;

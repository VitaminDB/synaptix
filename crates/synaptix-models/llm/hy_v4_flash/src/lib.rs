//! Hy4-flash (tencent, `hy_v4_flash`). Схема, эталон и сверка — `docs/hy4_flash.md`.

pub mod config;
pub mod dfly;
pub mod engram;
pub mod kern;
pub mod loader;
pub mod model;
pub mod pipeline;

pub use config::{AttnKind, Hy4Config};
pub use loader::{Hy4Weights, LoadError};
pub use model::{Hy4Model, ModelCache};
pub use pipeline::{Hy4Pipeline, Hy4Session, PipelineError};

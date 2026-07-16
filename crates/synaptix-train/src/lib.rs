#![allow(dead_code)]

pub mod adapters_train;
pub mod callbacks;
pub mod checkpointing;
pub mod distillation;
pub mod error;
pub mod eval;
pub mod losses;
pub mod metrics;
pub mod optimizer;
pub mod precision;
pub mod rlhf;
pub mod self_play;
pub mod trainer;

pub use error::{TrainError, Result};

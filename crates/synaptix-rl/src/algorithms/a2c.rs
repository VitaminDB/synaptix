use crate::error::Result;

pub struct A2cConfig { pub lr: f64, pub gamma: f64, pub clip_eps: f64 }
impl Default for A2cConfig { fn default() -> Self { Self { lr: 7e-4, gamma: 0.99, clip_eps: 0.5 } } }

pub fn compute_loss(_advantages: &[f32], _log_probs: &[f32]) -> Result<f32> { Ok(0.0) }

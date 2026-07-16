use crate::error::Result;

pub struct DqnConfig { pub lr: f64, pub gamma: f64, pub clip_eps: f64 }
impl Default for DqnConfig { fn default() -> Self { Self { lr: 1e-4, gamma: 0.99, clip_eps: 1.0 } } }

pub fn compute_loss(_q_values: &[f32], _targets: &[f32]) -> Result<f32> { Ok(0.0) }

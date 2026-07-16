use crate::error::Result;

pub struct SacConfig { pub lr: f64, pub gamma: f64, pub clip_eps: f64 }
impl Default for SacConfig { fn default() -> Self { Self { lr: 3e-4, gamma: 0.99, clip_eps: 0.2 } } }

pub fn compute_loss(_q1: &[f32], _q2: &[f32], _policy_log_probs: &[f32]) -> Result<f32> { Ok(0.0) }

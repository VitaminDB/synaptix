use crate::error::Result;

pub struct PpoConfig { pub lr: f64, pub gamma: f64, pub clip_eps: f64 }
impl Default for PpoConfig { fn default() -> Self { Self { lr: 3e-4, gamma: 0.99, clip_eps: 0.2 } } }

pub fn compute_loss(_advantages: &[f32], _log_probs: &[f32], _old_log_probs: &[f32]) -> Result<f32> { Ok(0.0) }

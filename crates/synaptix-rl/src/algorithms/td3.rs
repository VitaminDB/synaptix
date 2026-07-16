use crate::error::Result;

pub struct Td3Config { pub lr: f64, pub gamma: f64, pub clip_eps: f64 }
impl Default for Td3Config { fn default() -> Self { Self { lr: 3e-4, gamma: 0.99, clip_eps: 0.5 } } }

pub fn compute_loss(_q1: &[f32], _q2: &[f32], _policy_actions: &[f32]) -> Result<f32> { Ok(0.0) }

use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct PpoConfig { pub lr: f64 }
impl Default for PpoConfig { fn default() -> Self { Self { lr: 1e-5 } } }

pub fn compute_loss(logits: &Tensor) -> Result<Tensor> { Ok(logits.clone()) }

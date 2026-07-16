use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct GrpoConfig { pub lr: f64 }
impl Default for GrpoConfig { fn default() -> Self { Self { lr: 1e-5 } } }

pub fn compute_loss(logits: &Tensor) -> Result<Tensor> { Ok(logits.clone()) }

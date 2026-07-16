use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct OnPolicyConfig { pub temperature: f64 }
impl Default for OnPolicyConfig { fn default() -> Self { Self { temperature: 1.0 } } }

pub fn compute_loss(student: &Tensor, _teacher: &Tensor) -> Result<Tensor> { Ok(student.clone()) }

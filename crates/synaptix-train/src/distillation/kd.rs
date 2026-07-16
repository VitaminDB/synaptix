use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct KdConfig { pub temperature: f64 }
impl Default for KdConfig { fn default() -> Self { Self { temperature: 4.0 } } }

pub fn compute_loss(student: &Tensor, _teacher: &Tensor) -> Result<Tensor> { Ok(student.clone()) }

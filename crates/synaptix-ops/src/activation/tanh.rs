use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn tanh(x: &Tensor) -> Result<Tensor> { x.tanh() }

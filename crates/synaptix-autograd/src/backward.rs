use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn backward(output: &Tensor) -> Result<()> {
    synaptix_core::grad::backward(output)
}

pub fn backward_with(output: &Tensor, gradient: Tensor) -> Result<()> {
    synaptix_core::grad::backward_with(output, gradient)
}

pub fn grad(output: &Tensor, inputs: &[&Tensor]) -> Result<Vec<Tensor>> {
    synaptix_core::grad::backward(output)?;
    Ok(inputs
        .iter()
        .map(|t| t.grad().unwrap_or_else(|| t.zeros_like().unwrap_or_else(|_| (*t).clone())))
        .collect())
}

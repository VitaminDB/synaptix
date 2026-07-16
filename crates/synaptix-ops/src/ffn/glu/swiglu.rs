use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::activation::silu::silu;

pub fn swiglu(
    x: &Tensor,
    w_gate: &Tensor,
    w_up: &Tensor,
    w_down: &Tensor,
) -> Result<Tensor> {
    let gate = x.matmul(&w_gate.transpose(0, 1)?.contiguous()?)?;
    let up = x.matmul(&w_up.transpose(0, 1)?.contiguous()?)?;
    let activated = silu(&gate)?;
    let hidden = activated.mul(&up)?;
    hidden.matmul(&w_down.transpose(0, 1)?.contiguous()?)
}

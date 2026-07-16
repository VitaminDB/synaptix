use synaptix_core::{error::Result, tensor::Tensor};

use crate::activation::relu;

/// DeepSeek shared expert (всегда активный) — обычный 2-слойный FFN с ReLU:
///   `y = relu(x · fc1ᵀ) · fc2ᵀ`. `fc1:[H,D]`, `fc2:[D,H]`.
pub fn shared_expert_forward(x: &Tensor, fc1: &Tensor, fc2: &Tensor) -> Result<Tensor> {
    let h = x.matmul(&fc1.transpose(0, 1)?.contiguous()?)?;
    let h = relu(&h)?;
    h.matmul(&fc2.transpose(0, 1)?.contiguous()?)
}

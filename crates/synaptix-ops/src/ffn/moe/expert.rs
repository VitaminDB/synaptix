use synaptix_core::{error::Result, tensor::Tensor};

use crate::activation::relu;

/// Один эксперт MoE — стандартный 2-слойный FFN с ReLU:
///   `y = relu(x · fc1ᵀ) · fc2ᵀ`. `fc1:[H,D]`, `fc2:[D,H]` (конвенция nn.Linear).
pub struct Expert {
    pub fc1: Tensor,
    pub fc2: Tensor,
}

impl Expert {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = x.matmul(&self.fc1.transpose(0, 1)?.contiguous()?)?;
        let h = relu(&h)?;
        h.matmul(&self.fc2.transpose(0, 1)?.contiguous()?)
    }
}

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::dit::dit_block::{compute_adaln, modulate};
use crate::init::InitMethod;
use crate::linear::Linear;

pub struct AdaLnZero {
    pub modulation: Linear,
    pub hidden_size: usize,
}

impl AdaLnZero {
    pub fn new(hidden_size: usize, cond_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            modulation: Linear::from_init(cond_dim, 6 * hidden_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            hidden_size,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, hidden_size: usize) -> Result<Self> {
        Ok(Self {
            modulation: Linear::new(weight, bias)?,
            hidden_size,
        })
    }

    pub fn compute(
        &self,
        cond: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
        compute_adaln(&self.modulation, cond, self.hidden_size)
    }

    pub fn modulate(&self, x: &Tensor, shift: &Tensor, scale: &Tensor) -> Result<Tensor> {
        modulate(x, shift, scale)
    }
}

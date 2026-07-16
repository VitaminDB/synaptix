use crate::module::Module;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;

pub struct RewardHead {
    pub proj: Linear,
}

impl RewardHead {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, 1, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        Ok(Self { proj })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

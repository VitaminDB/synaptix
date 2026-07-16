use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct QaHead {
    pub proj: Linear,
}

impl QaHead {
    pub fn new(hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, 2, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        Ok(Self { proj: Linear::new(weight, bias)? })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }

    pub fn forward_split(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.proj.forward(x)?;
        let last = logits.rank() - 1;
        let start = logits.narrow(last, 0, 1)?.squeeze(last)?;
        let end = logits.narrow(last, 1, 1)?.squeeze(last)?;
        Ok((start, end))
    }
}

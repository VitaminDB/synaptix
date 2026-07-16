use crate::module::Module;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;

pub struct TokenClsHead {
    pub proj: Linear,
    pub num_labels: usize,
}

impl TokenClsHead {
    pub fn new(hidden_size: usize, num_labels: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, num_labels, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            num_labels,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let num_labels = proj.out_features();
        Ok(Self { proj, num_labels })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

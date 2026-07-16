use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::log_softmax_dim;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct CtcHead {
    pub proj: Linear,
    pub vocab_size: usize,
}

impl CtcHead {
    pub fn new(hidden_size: usize, vocab_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, vocab_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            vocab_size,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let vocab_size = proj.out_features();
        Ok(Self { proj, vocab_size })
    }

    pub fn logits(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.proj.forward(x)?;
        let last = logits.rank() - 1;
        log_softmax_dim(&logits, last)
    }
}

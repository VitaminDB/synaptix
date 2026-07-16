use crate::module::Module;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;

pub struct LmHead {
    pub proj: Linear,
    pub vocab_size: usize,
}

impl LmHead {
    pub fn new(hidden_size: usize, vocab_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, vocab_size, false, InitMethod::XavierNormal { fan_in: hidden_size, fan_out: vocab_size }, InitMethod::Zeros, device, dtype, 0)?,
            vocab_size,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let vocab_size = proj.out_features();
        Ok(Self { proj, vocab_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }

    pub fn forward_last(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let seq = dims[dims.len() - 2];
        let last = x.narrow(dims.len() - 2, seq - 1, 1)?;
        self.proj.forward(&last)
    }
}

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::layer_norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct MlmHead {
    pub dense: Linear,
    pub ln_weight: Option<Tensor>,
    pub ln_bias: Option<Tensor>,
    pub ln_eps: f32,
    pub out: Linear,
    pub vocab_size: usize,
}

impl MlmHead {
    pub fn new(hidden_size: usize, vocab_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            dense: Linear::from_init(hidden_size, hidden_size, true, InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 }, InitMethod::Zeros, device, dtype, 0)?,
            ln_weight: Some(Tensor::ones(&[hidden_size], dtype, device)?),
            ln_bias: Some(Tensor::zeros(&[hidden_size], dtype, device)?),
            ln_eps: 1e-12,
            out: Linear::from_init(hidden_size, vocab_size, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1)?,
            vocab_size,
        })
    }

    pub fn from_weights(
        dense_w: Tensor, dense_b: Option<Tensor>,
        ln_weight: Option<Tensor>, ln_bias: Option<Tensor>, ln_eps: f32,
        out_w: Tensor, out_b: Option<Tensor>,
    ) -> Result<Self> {
        let dense = Linear::new(dense_w, dense_b)?;
        let out_layer = Linear::new(out_w, out_b)?;
        let vocab_size = out_layer.out_features();
        Ok(Self { dense, ln_weight, ln_bias, ln_eps, out: out_layer, vocab_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dense.forward(x)?;
        let activated = h.gelu_exact()?;
        let normed = layer_norm(&activated, self.ln_weight.as_ref(), self.ln_bias.as_ref(), self.ln_eps)?;
        self.out.forward(&normed)
    }
}

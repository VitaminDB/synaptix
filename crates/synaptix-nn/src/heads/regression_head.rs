use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

#[derive(Clone, Copy, Debug)]
pub enum RegressionActivation {
    Tanh,
    GeluTanh,
    GeluExact,
    Relu,
    Silu,
    Identity,
}

pub struct RegressionHead {
    pub dense: Linear,
    pub activation: RegressionActivation,
    pub out: Linear,
    pub output_dim: usize,
}

impl RegressionHead {
    pub fn new(hidden_size: usize, output_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            dense: Linear::from_init(hidden_size, hidden_size, true, InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 }, InitMethod::Zeros, device, dtype, 0)?,
            activation: RegressionActivation::Tanh,
            out: Linear::from_init(hidden_size, output_dim, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1)?,
            output_dim,
        })
    }

    pub fn from_weights(
        dense_w: Tensor, dense_b: Option<Tensor>,
        out_w: Tensor, out_b: Option<Tensor>,
        activation: RegressionActivation,
    ) -> Result<Self> {
        let dense = Linear::new(dense_w, dense_b)?;
        let out_layer = Linear::new(out_w, out_b)?;
        let output_dim = out_layer.out_features();
        Ok(Self { dense, activation, out: out_layer, output_dim })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dense.forward(x)?;
        let activated = match self.activation {
            RegressionActivation::Tanh => h.tanh()?,
            RegressionActivation::GeluTanh => h.gelu_tanh()?,
            RegressionActivation::GeluExact => h.gelu_exact()?,
            RegressionActivation::Relu => h.relu()?,
            RegressionActivation::Silu => h.silu()?,
            RegressionActivation::Identity => h,
        };
        self.out.forward(&activated)
    }
}

use crate::module::Module;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::parameter::Parameter;

pub struct Ia3Linear {
    pub base: Linear,
    pub scale: Parameter,
}

impl Ia3Linear {
    pub fn new(in_features: usize, out_features: usize, device: Device, dtype: DType) -> Result<Self> {
        let scale_t = crate::init::init_tensor(&[out_features], InitMethod::Ones, dtype, 0, device)?;
        Ok(Self {
            base: Linear::from_init(in_features, out_features, false, InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 }, InitMethod::Zeros, device, dtype, 0)?,
            scale: Parameter::new(scale_t),
        })
    }

    pub fn from_weights(base_w: Tensor, scale: Tensor) -> Result<Self> {
        Ok(Self {
            base: Linear::new(base_w, None)?,
            scale: Parameter::new(scale),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.base.forward(x)?;
        y.broadcast_mul(&self.scale.tensor())
    }
}

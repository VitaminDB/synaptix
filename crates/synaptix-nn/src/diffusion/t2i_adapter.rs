use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// T2I-Adapter (Mou et al., 2023) — лёгкий аналог ControlNet'а: добавляет
/// adapter-фичи к выбранным residual-выходам UNet'а.
///
/// `y = x + scale · proj(condition)`.
pub struct T2iAdapter {
    pub proj: Linear,
    pub scale: f32,
}

impl T2iAdapter {
    pub fn new(in_channels: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            scale: 1.0,
        })
    }

    pub fn from_weights(proj_w: Tensor, proj_b: Option<Tensor>, scale: f32) -> Result<Self> {
        Ok(Self {
            proj: Linear::new(proj_w, proj_b)?,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, condition: &Tensor) -> Result<Tensor> {
        let projected = self.proj.forward(condition)?;
        let scaled = projected.affine(self.scale, 0.0)?;
        x.add(&scaled)
    }
}

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// ControlNet adapter (Zhang et al., 2023) — упрощённая residual-инъекция.
///
/// `y = x + conditioning_scale · proj(control)`.
///
/// `proj` инициализируется нулями (zero-init Linear), чтобы на старте обучения
/// ControlNet не возмущал предсказание базовой модели; bias тоже zero.
pub struct ControlNet {
    pub proj: Linear,
    pub conditioning_scale: f32,
}

impl ControlNet {
    pub fn new(in_channels: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0,
            )?,
            conditioning_scale: 1.0,
        })
    }

    pub fn from_weights(proj_w: Tensor, proj_b: Option<Tensor>, conditioning_scale: f32) -> Result<Self> {
        Ok(Self {
            proj: Linear::new(proj_w, proj_b)?,
            conditioning_scale,
        })
    }

    pub fn forward(&self, x: &Tensor, control: &Tensor) -> Result<Tensor> {
        let projected = self.proj.forward(control)?;
        let scaled = projected.affine(self.conditioning_scale, 0.0)?;
        x.add(&scaled)
    }
}

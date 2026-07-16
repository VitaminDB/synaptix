use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// GaLore — Gradient Low-Rank Projection.
///
/// Это **тренировочная техника** (оптимизатор проектирует градиент весов на
/// низкоранговое подпространство), архитектура forward-прохода не меняется.
/// На инференсе модуль эквивалентен обычному `Linear`; ранг/scale хранятся
/// только для целей сохранения совместимости загрузки.
pub struct GaloreLinear {
    pub base: Linear,
    pub rank: usize,
    pub scale: f32,
}

impl GaloreLinear {
    pub fn new(
        in_features: usize,
        out_features: usize,
        rank: usize,
        scale: f32,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            base: Linear::from_init(
                in_features, out_features, false,
                InitMethod::KaimingUniform { fan_in: in_features, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            rank,
            scale,
        })
    }

    pub fn from_weights(base_w: Tensor, rank: usize, scale: f32) -> Result<Self> {
        Ok(Self {
            base: Linear::new(base_w, None)?,
            rank,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.base.forward(x)
    }
}

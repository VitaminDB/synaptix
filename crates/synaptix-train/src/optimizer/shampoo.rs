use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct ShampooConfig { pub lr: f64 }
impl Default for ShampooConfig { fn default() -> Self { Self { lr: 1e-3 } } }

pub struct Shampoo { pub config: ShampooConfig }
impl Shampoo {
    pub fn new(config: ShampooConfig) -> Self { Self { config } }
    /// Не реализовано: шаг молча ничего не делал, обучение «шло» без
    /// обновления весов.
    pub fn step_params(&mut self, _params: &mut [Tensor], _grads: &[Tensor]) -> Result<()> {
        Err(crate::error::TrainError::Other("SHAMPOO: шаг оптимизатора не реализован".into()))
    }
}

use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct SoapConfig { pub lr: f64 }
impl Default for SoapConfig { fn default() -> Self { Self { lr: 1e-4 } } }

pub struct Soap { pub config: SoapConfig }
impl Soap {
    pub fn new(config: SoapConfig) -> Self { Self { config } }
    /// Не реализовано: шаг молча ничего не делал, обучение «шло» без
    /// обновления весов.
    pub fn step_params(&mut self, _params: &mut [Tensor], _grads: &[Tensor]) -> Result<()> {
        Err(crate::error::TrainError::Other("SOAP: шаг оптимизатора не реализован".into()))
    }
}

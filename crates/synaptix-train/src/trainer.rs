use crate::error::Result;
use synaptix_core::tensor::Tensor;

pub struct TrainerConfig {
    pub max_steps: usize,
    pub eval_steps: usize,
    pub save_steps: usize,
    pub log_steps: usize,
    pub gradient_accumulation_steps: usize,
    pub max_grad_norm: f64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self { max_steps: 1000, eval_steps: 100, save_steps: 500, log_steps: 10, gradient_accumulation_steps: 1, max_grad_norm: 1.0 }
    }
}

pub trait Trainer {
    fn train_step(&mut self, batch: &[Tensor]) -> Result<f64>;
    fn eval_step(&mut self, batch: &[Tensor]) -> Result<f64>;
    fn save_checkpoint(&self, path: &std::path::Path) -> Result<()>;
}

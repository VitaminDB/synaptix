use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{PredictionType, Scheduler, SchedulerOutput};

#[derive(Debug, Clone)]
pub struct FlowMatchConfig {
    pub n_train_timesteps: usize,
    pub shift: f32,
    pub use_dynamic_shifting: bool,
    pub base_shift: f32,
    pub max_shift: f32,
    pub base_seq_len: usize,
    pub max_seq_len: usize,
}

impl Default for FlowMatchConfig {
    fn default() -> Self {
        Self {
            n_train_timesteps: 1000,
            shift: 1.0,
            use_dynamic_shifting: false,
            base_shift: 0.5,
            max_shift: 1.15,
            base_seq_len: 256,
            max_seq_len: 4096,
        }
    }
}

impl FlowMatchConfig {
    pub fn sd3() -> Self {
        Self { shift: 3.0, ..Default::default() }
    }

    pub fn flux() -> Self {
        Self { use_dynamic_shifting: true, ..Default::default() }
    }
}

pub struct FlowMatchScheduler {
    cfg: FlowMatchConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl FlowMatchScheduler {
    pub fn new(cfg: FlowMatchConfig) -> Self {
        Self { cfg, sigmas: Vec::new(), timesteps: Vec::new() }
    }

    fn compute_shift(&self, seq_len: usize) -> f32 {
        if !self.cfg.use_dynamic_shifting {
            return self.cfg.shift;
        }
        let m = (self.cfg.max_shift - self.cfg.base_shift)
            / (self.cfg.max_seq_len as f32 - self.cfg.base_seq_len as f32);
        let b = self.cfg.base_shift - m * self.cfg.base_seq_len as f32;
        m * seq_len as f32 + b
    }

    pub fn set_timesteps_with_seq_len(&mut self, n_steps: usize, seq_len: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let shift = self.compute_shift(seq_len);
        let n = self.cfg.n_train_timesteps as f32;
        let denom = (n_steps - 1).max(1) as f32;
        let timesteps_raw: Vec<f32> = (0..n_steps)
            .map(|i| n - (n - 1.0) * (i as f32 / denom))
            .collect();
        let mut sigmas: Vec<f32> = timesteps_raw
            .iter()
            .map(|&t| {
                let t01 = t / n;
                shift * t01 / (1.0 + (shift - 1.0) * t01)
            })
            .collect();
        self.timesteps = sigmas.iter().map(|&s| s * n).collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;
        Ok(())
    }
}

impl Scheduler for FlowMatchScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        self.set_timesteps_with_seq_len(n_steps, self.cfg.base_seq_len)
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { PredictionType::FlowMatchVelocity }

    fn scale_model_input(&self, sample: &Tensor, _step_idx: usize) -> Result<Tensor> {
        Ok(sample.clone())
    }

    fn step(&mut self, model_output: &Tensor, step_idx: usize, sample: &Tensor) -> Result<SchedulerOutput> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange { idx: step_idx, n_steps: self.sigmas.len().saturating_sub(1) });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let dt = sigma_next - sigma;
        let prev_sample = sample.add(&model_output.affine(dt, 0.0)?)?;
        let x0 = sample.sub(&model_output.affine(sigma, 0.0)?)?;
        Ok(SchedulerOutput { prev_sample, pred_original_sample: Some(x0) })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        let x = original.affine(1.0 - sigma, 0.0)?;
        let n = noise.affine(sigma, 0.0)?;
        x.add(&n).map_err(DiffusionError::from)
    }
}

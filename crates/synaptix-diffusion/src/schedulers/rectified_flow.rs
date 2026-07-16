use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{PredictionType, Scheduler, SchedulerOutput};

#[derive(Debug, Clone)]
pub struct RectifiedFlowConfig {
    pub n_train_timesteps: usize,
    pub shift: f32,
}

impl Default for RectifiedFlowConfig {
    fn default() -> Self {
        Self { n_train_timesteps: 1000, shift: 1.0 }
    }
}

impl RectifiedFlowConfig {
    pub fn ltx() -> Self {
        Self { shift: 1.0, ..Default::default() }
    }
}

pub struct RectifiedFlowScheduler {
    cfg: RectifiedFlowConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
}

impl RectifiedFlowScheduler {
    pub fn new(cfg: RectifiedFlowConfig) -> Self {
        Self { cfg, sigmas: Vec::new(), timesteps: Vec::new() }
    }
}

impl Scheduler for RectifiedFlowScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n = self.cfg.n_train_timesteps;
        let shift = self.cfg.shift;
        let sigmas: Vec<f32> = (0..=n_steps)
            .rev()
            .map(|i| {
                let t = i as f32 / n_steps as f32;
                let s = shift * t / (1.0 + (shift - 1.0) * t);
                s.clamp(0.0, 1.0)
            })
            .collect();
        self.timesteps = sigmas[..n_steps].iter().map(|&s| s * n as f32).collect();
        self.sigmas = sigmas;
        if self.sigmas.last() != Some(&0.0) {
            self.sigmas.push(0.0);
        }
        Ok(())
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

use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{randn_like, PredictionType, Scheduler, SchedulerOutput};

#[derive(Debug, Clone)]
pub struct ConsistencyConfig {
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub rho: f32,
    pub seed: u64,
}

impl Default for ConsistencyConfig {
    fn default() -> Self {
        Self { sigma_min: 0.002, sigma_max: 80.0, rho: 7.0, seed: 0 }
    }
}

pub struct ConsistencyScheduler {
    cfg: ConsistencyConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    rng: Philox4x32,
}

impl ConsistencyScheduler {
    pub fn new(cfg: ConsistencyConfig) -> Self {
        let rng = Philox4x32::new(cfg.seed);
        Self { cfg, sigmas: Vec::new(), timesteps: Vec::new(), rng }
    }
}

impl Scheduler for ConsistencyScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let inv_rho = 1.0 / self.cfg.rho;
        let smax = self.cfg.sigma_max.powf(inv_rho);
        let smin = self.cfg.sigma_min.powf(inv_rho);
        let mut sigmas = Vec::with_capacity(n_steps + 1);
        for i in 0..n_steps {
            let r = i as f32 / (n_steps - 1).max(1) as f32;
            sigmas.push((smax + r * (smin - smax)).powf(self.cfg.rho));
        }
        sigmas.push(0.0);
        self.sigmas = sigmas.clone();
        self.timesteps = sigmas[..n_steps].to_vec();
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { PredictionType::SampleX0 }

    fn step(&mut self, model_output: &Tensor, step_idx: usize, _sample: &Tensor) -> Result<SchedulerOutput> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange { idx: step_idx, n_steps: self.sigmas.len().saturating_sub(1) });
        }
        let sigma_next = self.sigmas[step_idx + 1];
        let x0 = model_output.clone();
        let prev_sample = if sigma_next > 0.0 {
            let noise = randn_like(&x0, &mut self.rng)?;
            x0.add(&noise.affine(sigma_next, 0.0)?)?
        } else {
            x0.clone()
        };
        Ok(SchedulerOutput { prev_sample, pred_original_sample: Some(x0) })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        crate::schedulers::add_noise_ve(original, noise, sigma)
    }

    fn reset_state(&mut self) { self.rng = Philox4x32::new(self.cfg.seed); }
}

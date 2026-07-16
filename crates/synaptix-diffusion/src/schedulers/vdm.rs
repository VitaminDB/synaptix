use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    convert_to_x0, randn_like, PredictionType, Scheduler, SchedulerOutput,
};

#[derive(Debug, Clone)]
pub struct VdmConfig {
    pub gamma_min: f32,
    pub gamma_max: f32,
    pub prediction_type: PredictionType,
    pub seed: u64,
}

impl Default for VdmConfig {
    fn default() -> Self {
        Self {
            gamma_min: -13.3,
            gamma_max: 5.0,
            prediction_type: PredictionType::Epsilon,
            seed: 0,
        }
    }
}

pub struct VdmScheduler {
    cfg: VdmConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    rng: Philox4x32,
}

impl VdmScheduler {
    pub fn new(cfg: VdmConfig) -> Self {
        let rng = Philox4x32::new(cfg.seed);
        Self { cfg, sigmas: Vec::new(), timesteps: Vec::new(), rng }
    }

    fn gamma_to_sigma(&self, gamma: f32) -> f32 {
        (-gamma / 2.0).exp().sqrt()
    }

    pub fn gamma_to_alpha(&self, gamma: f32) -> f32 {
        1.0 / (1.0 + self.gamma_to_sigma(gamma).powi(2)).sqrt()
    }
}

impl Scheduler for VdmScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let gmax = self.cfg.gamma_max;
        let gmin = self.cfg.gamma_min;
        let mut sigmas = Vec::with_capacity(n_steps + 1);
        for i in 0..n_steps {
            let r = i as f32 / (n_steps - 1).max(1) as f32;
            let g = gmax + r * (gmin - gmax);
            sigmas.push(self.gamma_to_sigma(g));
        }
        sigmas.push(0.0);
        self.sigmas = sigmas.clone();
        self.timesteps = sigmas[..n_steps].to_vec();
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { self.cfg.prediction_type }

    fn step(&mut self, model_output: &Tensor, step_idx: usize, sample: &Tensor) -> Result<SchedulerOutput> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange { idx: step_idx, n_steps: self.sigmas.len().saturating_sub(1) });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let x0 = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;
        let alpha = crate::schedulers::sigma_to_alpha(sigma);
        let alpha_next = crate::schedulers::sigma_to_alpha(sigma_next);
        let prev_sample = if sigma_next <= 0.0 {
            x0.clone()
        } else {
            let variance = 1.0 - (alpha_next / alpha).powi(2);
            let noise = randn_like(&x0, &mut self.rng)?;
            x0.affine(alpha_next, 0.0)?
                .add(&noise.affine(variance.max(0.0).sqrt(), 0.0)?)?
        };
        Ok(SchedulerOutput { prev_sample, pred_original_sample: Some(x0) })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        crate::schedulers::add_noise_vp(original, noise, sigma)
    }

    fn reset_state(&mut self) { self.rng = Philox4x32::new(self.cfg.seed); }
}

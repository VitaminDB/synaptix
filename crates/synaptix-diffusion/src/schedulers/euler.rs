use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, alphas_to_sigmas, betas_for, convert_to_x0, karras_sigmas,
    timesteps_from_spacing, BetaConfig, PredictionType, Scheduler, SchedulerOutput,
    TimestepSpacing,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SigmaSchedule {
    BetaSchedule,
    Karras,
    Exponential,
}

#[derive(Debug, Clone)]
pub struct EulerConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub sigma_schedule: SigmaSchedule,
    pub karras_rho: f32,
    pub use_scale_model_input: bool,
}

impl Default for EulerConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Leading,
            sigma_schedule: SigmaSchedule::BetaSchedule,
            karras_rho: 7.0,
            use_scale_model_input: true,
        }
    }
}

pub struct EulerScheduler {
    cfg: EulerConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
}

impl EulerScheduler {
    pub fn new(cfg: EulerConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
        }
    }

    pub fn sigma(&self, step_idx: usize) -> f32 {
        self.sigmas.get(step_idx).copied().unwrap_or(0.0)
    }

    pub fn override_sigmas(&mut self, sigmas: Vec<f32>) {
        self.sigmas = sigmas;
    }
}

impl Scheduler for EulerScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        self.timestep_indices = timesteps_from_spacing(n_train, n_steps, self.cfg.spacing, 0);
        self.timesteps = self.timestep_indices.iter().map(|&i| i as f32).collect();

        match self.cfg.sigma_schedule {
            SigmaSchedule::BetaSchedule => {
                let sigmas_full: Vec<f32> = self
                    .alphas_cum
                    .iter()
                    .map(|&a| {
                        let a = a.clamp(1e-12, 1.0 - 1e-12);
                        ((1.0 - a) / a).sqrt()
                    })
                    .collect();
                let timesteps_float: Vec<f32> = match self.cfg.spacing {
                    TimestepSpacing::Linspace => {
                        let last = (n_train - 1) as f32;
                        let denom = (n_steps - 1).max(1) as f32;
                        (0..n_steps)
                            .map(|i| last * (1.0 - i as f32 / denom))
                            .collect()
                    }
                    _ => self.timestep_indices.iter().map(|&i| i as f32).collect(),
                };
                let mut sigmas = Vec::with_capacity(timesteps_float.len() + 1);
                let n_full = sigmas_full.len();
                for &t in &timesteps_float {
                    let low = (t.floor() as i64).max(0) as usize;
                    let high = (low + 1).min(n_full - 1);
                    let frac = t - low as f32;
                    sigmas.push(sigmas_full[low] * (1.0 - frac) + sigmas_full[high] * frac);
                }
                if matches!(self.cfg.spacing, TimestepSpacing::Linspace) {
                    self.timesteps = timesteps_float;
                }
                sigmas.push(0.0);
                self.sigmas = sigmas;
            }
            SigmaSchedule::Karras => {
                let all = alphas_to_sigmas(&self.alphas_cum);
                let smin = *all.first().unwrap();
                let smax = *all.last().unwrap();
                self.sigmas = karras_sigmas(smin, smax, n_steps, self.cfg.karras_rho);
            }
            SigmaSchedule::Exponential => {
                let all = alphas_to_sigmas(&self.alphas_cum);
                let smin = *all.first().unwrap();
                let smax = *all.last().unwrap();
                self.sigmas = crate::schedulers::exponential_sigmas(smin, smax, n_steps);
            }
        }
        Ok(())
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    fn prediction_type(&self) -> PredictionType {
        self.cfg.prediction_type
    }

    fn scale_model_input(&self, sample: &Tensor, step_idx: usize) -> Result<Tensor> {
        if !self.cfg.use_scale_model_input {
            return Ok(sample.clone());
        }
        let sigma = self.sigma(step_idx);
        let s = 1.0 / (sigma * sigma + 1.0).sqrt();
        sample.affine(s, 0.0).map_err(DiffusionError::from)
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
    ) -> Result<SchedulerOutput> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange {
                idx: step_idx,
                n_steps: self.sigmas.len().saturating_sub(1),
            });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let x0 = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;
        let derivative = sample.sub(&x0)?.affine(1.0 / sigma.max(1e-12), 0.0)?;
        let dt = sigma_next - sigma;
        let prev_sample = sample.add(&derivative.affine(dt, 0.0)?)?;
        Ok(SchedulerOutput {
            prev_sample,
            pred_original_sample: Some(x0),
        })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigma(step_idx);
        crate::schedulers::add_noise_ve(original, noise, sigma)
    }
}

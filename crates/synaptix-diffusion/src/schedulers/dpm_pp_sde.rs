use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::euler::SigmaSchedule;
use crate::schedulers::{
    alphas_cumprod, alphas_to_sigmas, betas_for, convert_to_x0, exponential_sigmas, karras_sigmas,
    randn_like, timesteps_from_spacing, BetaConfig, PredictionType, Scheduler, SchedulerOutput,
    TimestepSpacing,
};

#[derive(Debug, Clone)]
pub struct DpmPpSdeConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub sigma_schedule: SigmaSchedule,
    pub karras_rho: f32,
    pub eta: f32,
    pub r: f32,
    pub use_scale_model_input: bool,
    pub seed: u64,
}

impl Default for DpmPpSdeConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Trailing,
            sigma_schedule: SigmaSchedule::Karras,
            karras_rho: 7.0,
            eta: 1.0,
            r: 0.5,
            use_scale_model_input: true,
            seed: 0,
        }
    }
}

pub struct DpmPpSdeScheduler {
    cfg: DpmPpSdeConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    denoised_prev: Option<Tensor>,
    h_last: Option<f32>,
    rng: Philox4x32,
}

impl DpmPpSdeScheduler {
    pub fn new(cfg: DpmPpSdeConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        let rng = Philox4x32::new(cfg.seed);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
            denoised_prev: None,
            h_last: None,
            rng,
        }
    }
}

impl Scheduler for DpmPpSdeScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        self.timestep_indices = timesteps_from_spacing(n_train, n_steps, self.cfg.spacing, 0);
        self.timesteps = self.timestep_indices.iter().map(|&i| i as f32).collect();
        match self.cfg.sigma_schedule {
            SigmaSchedule::BetaSchedule => {
                let mut sigmas: Vec<f32> = self
                    .timestep_indices
                    .iter()
                    .map(|&i| {
                        let a = self.alphas_cum[i].clamp(1e-12, 1.0 - 1e-12);
                        ((1.0 - a) / a).sqrt()
                    })
                    .collect();
                sigmas.push(0.0);
                self.sigmas = sigmas;
            }
            SigmaSchedule::Karras => {
                let all = alphas_to_sigmas(&self.alphas_cum);
                self.sigmas = karras_sigmas(
                    *all.first().unwrap(),
                    *all.last().unwrap(),
                    n_steps,
                    self.cfg.karras_rho,
                );
            }
            SigmaSchedule::Exponential => {
                let all = alphas_to_sigmas(&self.alphas_cum);
                self.sigmas =
                    exponential_sigmas(*all.first().unwrap(), *all.last().unwrap(), n_steps);
            }
        }
        self.denoised_prev = None;
        self.h_last = None;
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { self.cfg.prediction_type }

    fn scale_model_input(&self, sample: &Tensor, step_idx: usize) -> Result<Tensor> {
        if !self.cfg.use_scale_model_input {
            return Ok(sample.clone());
        }
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
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
        let denoised = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;

        let prev_sample = if sigma_next <= 0.0 {
            denoised.clone()
        } else {
            let t = -sigma.max(1e-12).ln();
            let s = -sigma_next.max(1e-12).ln();
            let h = s - t;
            let eta_h = self.cfg.eta * h;
            let scale = (-eta_h).exp() * (sigma_next / sigma);
            let weight = (1.0 - (-h - eta_h).exp()) * (-eta_h).exp();

            let denoised_eff = match (&self.denoised_prev, self.h_last) {
                (Some(prev), Some(h_last)) if h_last > 1e-8 => {
                    let r = h_last / h;
                    let c_cur = 1.0 + 1.0 / (2.0 * r);
                    let c_prev = -1.0 / (2.0 * r);
                    denoised.affine(c_cur, 0.0)?.add(&prev.affine(c_prev, 0.0)?)?
                }
                _ => denoised.clone(),
            };

            let noise_sigma = sigma_next * ((-2.0 * eta_h).exp() - 1.0).max(0.0).sqrt();
            let mut prev = sample.affine(scale, 0.0)?.add(&denoised_eff.affine(weight, 0.0)?)?;
            if noise_sigma > 0.0 {
                let noise = randn_like(&prev, &mut self.rng)?;
                prev = prev.add(&noise.affine(noise_sigma, 0.0)?)?;
            }
            prev
        };

        self.denoised_prev = Some(denoised.clone());
        self.h_last = if sigma_next > 0.0 {
            let h = (-sigma.max(1e-12).ln()) - (-sigma_next.max(1e-12).ln());
            Some(h)
        } else {
            None
        };
        Ok(SchedulerOutput {
            prev_sample,
            pred_original_sample: Some(denoised),
        })
    }

    fn step_with_noise(
        &mut self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
        noise: &Tensor,
    ) -> Result<SchedulerOutput> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange {
                idx: step_idx,
                n_steps: self.sigmas.len().saturating_sub(1),
            });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let denoised = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;

        let prev_sample = if sigma_next <= 0.0 {
            denoised.clone()
        } else {
            let h = (-sigma.max(1e-12).ln()) - (-sigma_next.max(1e-12).ln());
            let eta_h = self.cfg.eta * h;
            let scale = (-eta_h).exp() * (sigma_next / sigma);
            let weight = (1.0 - (-h - eta_h).exp()) * (-eta_h).exp();
            let denoised_eff = match (&self.denoised_prev, self.h_last) {
                (Some(prev), Some(h_last)) if h_last > 1e-8 => {
                    let r = h_last / h;
                    let c_cur = 1.0 + 1.0 / (2.0 * r);
                    let c_prev = -1.0 / (2.0 * r);
                    denoised.affine(c_cur, 0.0)?.add(&prev.affine(c_prev, 0.0)?)?
                }
                _ => denoised.clone(),
            };
            let noise_sigma = sigma_next * ((-2.0 * eta_h).exp() - 1.0).max(0.0).sqrt();
            let mut prev = sample.affine(scale, 0.0)?.add(&denoised_eff.affine(weight, 0.0)?)?;
            if noise_sigma > 0.0 {
                prev = prev.add(&noise.affine(noise_sigma, 0.0)?)?;
            }
            prev
        };

        self.denoised_prev = Some(denoised.clone());
        self.h_last = if sigma_next > 0.0 {
            Some((-sigma.max(1e-12).ln()) - (-sigma_next.max(1e-12).ln()))
        } else {
            None
        };
        Ok(SchedulerOutput {
            prev_sample,
            pred_original_sample: Some(denoised),
        })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        crate::schedulers::add_noise_ve(original, noise, sigma)
    }

    fn reset_state(&mut self) {
        self.denoised_prev = None;
        self.h_last = None;
        self.rng = Philox4x32::new(self.cfg.seed);
    }
}

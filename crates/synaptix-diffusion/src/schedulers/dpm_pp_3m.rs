use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::euler::SigmaSchedule;
use crate::schedulers::{
    alphas_cumprod, alphas_to_sigmas, betas_for, convert_to_x0, exponential_sigmas, karras_sigmas,
    timesteps_from_spacing, BetaConfig, PredictionType, Scheduler, SchedulerOutput,
    TimestepSpacing,
};

#[derive(Debug, Clone)]
pub struct DpmPp3MConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub sigma_schedule: SigmaSchedule,
    pub karras_rho: f32,
    pub use_scale_model_input: bool,
}

impl Default for DpmPp3MConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Trailing,
            sigma_schedule: SigmaSchedule::Karras,
            karras_rho: 7.0,
            use_scale_model_input: true,
        }
    }
}

pub struct DpmPp3MScheduler {
    cfg: DpmPp3MConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    history: Vec<Tensor>,
    h_history: Vec<f32>,
}

impl DpmPp3MScheduler {
    pub fn new(cfg: DpmPp3MConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
            history: Vec::new(),
            h_history: Vec::new(),
        }
    }
}

impl Scheduler for DpmPp3MScheduler {
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
        self.history.clear();
        self.h_history.clear();
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
            let h = (-sigma.max(1e-12).ln()) - (-sigma_next.max(1e-12).ln());
            let scale = (-h).exp();
            let w0 = 1.0 - (-h).exp();

            let denoised_eff = match self.history.len() {
                0 => denoised.clone(),
                1 => {
                    let h1 = self.h_history[0];
                    let r1 = h1 / h;
                    denoised.affine(1.0 + 1.0 / (2.0 * r1), 0.0)?
                        .sub(&self.history[0].affine(1.0 / (2.0 * r1), 0.0)?)?
                }
                _ => {
                    let h1 = *self.h_history.last().unwrap();
                    let h2 = self.h_history[self.h_history.len().saturating_sub(2)];
                    let r1 = h1 / h;
                    let r2 = h2 / h;
                    let d0 = &denoised;
                    let d1 = &self.history[self.history.len() - 1];
                    let d2 = &self.history[self.history.len().saturating_sub(2)];
                    let a0 = 1.0 + 1.0 / (2.0 * r1) + 1.0 / (3.0 * r1 * r2);
                    let a1 = -1.0 / (2.0 * r1) - 1.0 / (2.0 * r1 * r2) - 1.0 / (3.0 * r1 * r2);
                    let a2 = 1.0 / (6.0 * r1 * r2);
                    d0.affine(a0, 0.0)?
                        .add(&d1.affine(a1, 0.0)?)?
                        .add(&d2.affine(a2, 0.0)?)?
                }
            };

            let prev = sample.affine(scale, 0.0)?.add(&denoised_eff.affine(w0, 0.0)?)?;

            if self.history.len() == 3 {
                self.history.remove(0);
                self.h_history.remove(0);
            }
            self.history.push(denoised.clone());
            self.h_history.push(h);
            return Ok(SchedulerOutput {
                prev_sample: prev,
                pred_original_sample: Some(denoised),
            });
        };

        if self.history.len() == 3 {
            self.history.remove(0);
            self.h_history.remove(0);
        }
        let h = (-sigma.max(1e-12).ln()) - (-sigma_next.max(1e-12).ln()).max(-sigma.max(1e-12).ln());
        self.history.push(denoised.clone());
        self.h_history.push(h.abs().max(1e-8));

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
        self.history.clear();
        self.h_history.clear();
    }
}

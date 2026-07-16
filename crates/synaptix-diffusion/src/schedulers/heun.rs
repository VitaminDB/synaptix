use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::euler::SigmaSchedule;
use crate::schedulers::{
    alphas_cumprod, alphas_to_sigmas, betas_for, convert_to_x0, exponential_sigmas, karras_sigmas,
    timesteps_from_spacing, BetaConfig, PredictionType, Scheduler, SchedulerOutput,
    TimestepSpacing,
};

#[derive(Debug, Clone)]
pub struct HeunConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub sigma_schedule: SigmaSchedule,
    pub karras_rho: f32,
    pub use_scale_model_input: bool,
}

impl Default for HeunConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Leading,
            sigma_schedule: SigmaSchedule::Karras,
            karras_rho: 7.0,
            use_scale_model_input: true,
        }
    }
}

pub struct HeunScheduler {
    cfg: HeunConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    pending_correction: Option<HeunCorrection>,
}

struct HeunCorrection {
    sample_before: Tensor,
    d: Tensor,
    sigma: f32,
    sigma_next: f32,
}

impl HeunScheduler {
    pub fn new(cfg: HeunConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
            pending_correction: None,
        }
    }
}

impl Scheduler for HeunScheduler {
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
        self.pending_correction = None;
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { self.cfg.prediction_type }

    fn scale_model_input(&self, sample: &Tensor, step_idx: usize) -> Result<Tensor> {
        if !self.cfg.use_scale_model_input {
            return Ok(sample.clone());
        }
        let sigma = if let Some(pc) = &self.pending_correction {
            pc.sigma_next
        } else {
            self.sigmas.get(step_idx).copied().unwrap_or(0.0)
        };
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
        if let Some(pc) = self.pending_correction.take() {
            let x0 = convert_to_x0(sample, model_output, pc.sigma_next, self.cfg.prediction_type)?;
            let d2 = sample.sub(&x0)?.affine(1.0 / pc.sigma_next.max(1e-12), 0.0)?;
            let d_avg = pc.d.add(&d2)?.affine(0.5, 0.0)?;
            let dt = pc.sigma_next - pc.sigma;
            let prev_sample = pc.sample_before.add(&d_avg.affine(dt, 0.0)?)?;
            return Ok(SchedulerOutput {
                prev_sample,
                pred_original_sample: Some(x0),
            });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let x0 = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;
        let d = sample.sub(&x0)?.affine(1.0 / sigma.max(1e-12), 0.0)?;
        let dt = sigma_next - sigma;
        let euler_prev = sample.add(&d.affine(dt, 0.0)?)?;
        if sigma_next <= 0.0 {
            return Ok(SchedulerOutput {
                prev_sample: euler_prev,
                pred_original_sample: Some(x0),
            });
        }
        self.pending_correction = Some(HeunCorrection {
            sample_before: sample.clone(),
            d,
            sigma,
            sigma_next,
        });
        Ok(SchedulerOutput {
            prev_sample: euler_prev,
            pred_original_sample: Some(x0),
        })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        crate::schedulers::add_noise_ve(original, noise, sigma)
    }

    fn reset_state(&mut self) {
        self.pending_correction = None;
    }
}

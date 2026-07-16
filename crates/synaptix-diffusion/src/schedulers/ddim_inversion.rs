use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, convert_to_eps, convert_to_x0, timesteps_from_spacing, BetaConfig,
    PredictionType, Scheduler, SchedulerOutput, TimestepSpacing,
};

#[derive(Debug, Clone)]
pub struct DdimInversionConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
}

impl Default for DdimInversionConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Leading,
        }
    }
}

pub struct DdimInversionScheduler {
    cfg: DdimInversionConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
}

impl DdimInversionScheduler {
    pub fn new(cfg: DdimInversionConfig) -> Self {
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
}

impl Scheduler for DdimInversionScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        let mut idxs = timesteps_from_spacing(n_train, n_steps, self.cfg.spacing, 0);
        idxs.reverse();
        self.timestep_indices = idxs;
        self.timesteps = self.timestep_indices.iter().map(|&i| i as f32).collect();
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

    fn step(
        &mut self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
    ) -> Result<SchedulerOutput> {
        if step_idx >= self.timestep_indices.len() {
            return Err(DiffusionError::StepOutOfRange {
                idx: step_idx,
                n_steps: self.timestep_indices.len(),
            });
        }
        let t_idx = self.timestep_indices[step_idx];
        let alpha_bar_t = self.alphas_cum[t_idx];
        let alpha_bar_next = if step_idx + 1 < self.timestep_indices.len() {
            self.alphas_cum[self.timestep_indices[step_idx + 1]]
        } else {
            self.alphas_cum[self.alphas_cum.len() - 1]
        };
        let sigma_t = ((1.0 - alpha_bar_t) / alpha_bar_t.max(1e-12)).sqrt();
        let x0 = convert_to_x0(sample, model_output, sigma_t, self.cfg.prediction_type)?;
        let eps = convert_to_eps(sample, model_output, sigma_t, self.cfg.prediction_type)?;
        let coef_x0 = alpha_bar_next.max(0.0).sqrt();
        let coef_eps = (1.0 - alpha_bar_next).max(0.0).sqrt();
        let next_sample = x0
            .affine(coef_x0, 0.0)?
            .add(&eps.affine(coef_eps, 0.0)?)?;
        Ok(SchedulerOutput {
            prev_sample: next_sample,
            pred_original_sample: Some(x0),
        })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let idx = step_idx.min(self.timestep_indices.len().saturating_sub(1));
        let t_idx = self.timestep_indices[idx];
        let alpha_bar = self.alphas_cum[t_idx];
        let sqrt_alpha = alpha_bar.max(1e-12).sqrt();
        let sqrt_one_minus = (1.0 - alpha_bar).max(0.0).sqrt();
        let a = original.affine(sqrt_alpha, 0.0)?;
        let b = noise.affine(sqrt_one_minus, 0.0)?;
        a.add(&b).map_err(DiffusionError::from)
    }
}

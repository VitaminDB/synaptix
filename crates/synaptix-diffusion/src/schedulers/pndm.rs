use std::collections::VecDeque;

use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, convert_to_eps, convert_to_x0, timesteps_from_spacing, BetaConfig,
    PredictionType, Scheduler, SchedulerOutput, TimestepSpacing,
};

#[derive(Debug, Clone)]
pub struct PndmConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub skip_prk_steps: bool,
    pub set_alpha_to_one: bool,
}

impl Default for PndmConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Leading,
            skip_prk_steps: true,
            set_alpha_to_one: true,
        }
    }
}

pub struct PndmScheduler {
    cfg: PndmConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    eps_history: VecDeque<Tensor>,
    last_sample: Option<Tensor>,
}

impl PndmScheduler {
    pub fn new(cfg: PndmConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
            eps_history: VecDeque::with_capacity(4),
            last_sample: None,
        }
    }

    fn alpha_bar_prev(&self, step_idx: usize) -> f32 {
        if step_idx + 1 < self.timestep_indices.len() {
            self.alphas_cum[self.timestep_indices[step_idx + 1]]
        } else if self.cfg.set_alpha_to_one {
            1.0
        } else {
            self.alphas_cum[0]
        }
    }

    fn ddim_step(
        &self,
        sample: &Tensor,
        eps: &Tensor,
        alpha_bar_t: f32,
        alpha_bar_prev: f32,
    ) -> Result<Tensor> {
        let one_minus = (1.0 - alpha_bar_t).max(0.0).sqrt();
        let sample_clean = sample
            .sub(&eps.affine(one_minus, 0.0)?)?
            .affine(1.0 / alpha_bar_t.max(1e-12).sqrt(), 0.0)?;
        let coef_x0 = alpha_bar_prev.max(0.0).sqrt();
        let coef_eps = (1.0 - alpha_bar_prev).max(0.0).sqrt();
        sample_clean
            .affine(coef_x0, 0.0)?
            .add(&eps.affine(coef_eps, 0.0)?)
            .map_err(DiffusionError::from)
    }
}

impl Scheduler for PndmScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        self.timestep_indices = timesteps_from_spacing(n_train, n_steps, self.cfg.spacing, 0);
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
        self.eps_history.clear();
        self.last_sample = None;
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
        let alpha_bar_prev = self.alpha_bar_prev(step_idx);
        let sigma_t = ((1.0 - alpha_bar_t) / alpha_bar_t.max(1e-12)).sqrt();

        let eps = convert_to_eps(sample, model_output, sigma_t, self.cfg.prediction_type)?;
        let x0 = convert_to_x0(sample, model_output, sigma_t, self.cfg.prediction_type)?;

        let prev_sample = if self.eps_history.len() < 3 && !self.cfg.skip_prk_steps {
            self.ddim_step(sample, &eps, alpha_bar_t, alpha_bar_prev)?
        } else {
            let blended = match self.eps_history.len() {
                0 => eps.clone(),
                1 => {
                    let last = self.eps_history.back().unwrap();
                    eps.affine(0.5, 0.0)?.add(&last.affine(0.5, 0.0)?)?
                }
                2 => {
                    let mut iter = self.eps_history.iter().rev();
                    let l1 = iter.next().unwrap();
                    let l2 = iter.next().unwrap();
                    eps.affine(23.0 / 12.0, 0.0)?
                        .sub(&l1.affine(16.0 / 12.0, 0.0)?)?
                        .add(&l2.affine(5.0 / 12.0, 0.0)?)?
                }
                _ => {
                    let mut iter = self.eps_history.iter().rev();
                    let l1 = iter.next().unwrap();
                    let l2 = iter.next().unwrap();
                    let l3 = iter.next().unwrap();
                    eps.affine(55.0 / 24.0, 0.0)?
                        .sub(&l1.affine(59.0 / 24.0, 0.0)?)?
                        .add(&l2.affine(37.0 / 24.0, 0.0)?)?
                        .sub(&l3.affine(9.0 / 24.0, 0.0)?)?
                }
            };
            let source = self.last_sample.as_ref().unwrap_or(sample).clone();
            self.ddim_step(&source, &blended, alpha_bar_t, alpha_bar_prev)?
        };

        self.last_sample = Some(sample.clone());
        if self.eps_history.len() == 4 {
            self.eps_history.pop_front();
        }
        self.eps_history.push_back(eps);

        Ok(SchedulerOutput {
            prev_sample,
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

    fn reset_state(&mut self) {
        self.eps_history.clear();
        self.last_sample = None;
    }
}

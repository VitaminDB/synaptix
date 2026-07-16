use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, randn_like, timesteps_from_spacing,
    BetaConfig, PredictionType, Scheduler, SchedulerOutput, TimestepSpacing,
};

pub(crate) fn ddpm_x0_and_eps(
    sample: &Tensor,
    model_output: &Tensor,
    alpha_bar_t: f32,
    prediction_type: PredictionType,
) -> Result<(Tensor, Tensor)> {
    let sqrt_alpha = alpha_bar_t.max(1e-12).sqrt();
    let sqrt_one_minus_alpha = (1.0 - alpha_bar_t).max(0.0).sqrt();
    match prediction_type {
        PredictionType::Epsilon => {
            let x0 = sample
                .sub(&model_output.affine(sqrt_one_minus_alpha, 0.0)?)?
                .affine(1.0 / sqrt_alpha, 0.0)?;
            Ok((x0, model_output.clone()))
        }
        PredictionType::SampleX0 => {
            let eps = sample
                .sub(&model_output.affine(sqrt_alpha, 0.0)?)?
                .affine(1.0 / sqrt_one_minus_alpha.max(1e-12), 0.0)?;
            Ok((model_output.clone(), eps))
        }
        PredictionType::Velocity => {
            let x0 = sample
                .affine(sqrt_alpha, 0.0)?
                .sub(&model_output.affine(sqrt_one_minus_alpha, 0.0)?)?;
            let eps = sample
                .affine(sqrt_one_minus_alpha, 0.0)?
                .add(&model_output.affine(sqrt_alpha, 0.0)?)?;
            Ok((x0, eps))
        }
        PredictionType::FlowMatchVelocity => Err(DiffusionError::invalid_arg(
            "ddpm_x0_and_eps: FlowMatchVelocity не поддерживается DDPM-параметризацией",
        )),
    }
}

#[derive(Debug, Clone)]
pub struct DdimConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub eta: f32,
    pub clip_sample: Option<f32>,
    pub set_alpha_to_one: bool,
    pub seed: u64,
}

impl Default for DdimConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Leading,
            eta: 0.0,
            clip_sample: None,
            set_alpha_to_one: true,
            seed: 0,
        }
    }
}

pub struct DdimScheduler {
    cfg: DdimConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    rng: Philox4x32,
}

impl DdimScheduler {
    pub fn new(cfg: DdimConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        let rng = Philox4x32::new(cfg.seed);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
            rng,
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
}

impl Scheduler for DdimScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        if n_steps > n_train {
            return Err(DiffusionError::invalid_arg(format!(
                "n_steps={n_steps} > num_train_timesteps={n_train}"
            )));
        }
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
        Ok(())
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    fn init_noise_sigma(&self) -> f32 {
        1.0
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

        let (x0, eps) = ddpm_x0_and_eps(
            sample,
            model_output,
            alpha_bar_t,
            self.cfg.prediction_type,
        )?;
        let x0 = if let Some(c) = self.cfg.clip_sample {
            x0.clamp(-c, c)?
        } else {
            x0
        };

        let variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).max(1e-12)
            * (1.0 - alpha_bar_t / alpha_bar_prev.max(1e-12));
        let stddev = self.cfg.eta * variance.max(0.0).sqrt();
        let coef_x0 = alpha_bar_prev.max(0.0).sqrt();
        let coef_eps = (1.0 - alpha_bar_prev - stddev * stddev).max(0.0).sqrt();
        let prev_mean = x0
            .affine(coef_x0, 0.0)?
            .add(&eps.affine(coef_eps, 0.0)?)?;
        let prev_sample = if stddev > 0.0 {
            let noise = randn_like(&prev_mean, &mut self.rng)?;
            prev_mean.add(&noise.affine(stddev, 0.0)?)?
        } else {
            prev_mean
        };
        Ok(SchedulerOutput {
            prev_sample,
            pred_original_sample: Some(x0),
        })
    }

    fn step_with_noise(
        &mut self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
        noise: &Tensor,
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
        let (x0, eps) = ddpm_x0_and_eps(
            sample,
            model_output,
            alpha_bar_t,
            self.cfg.prediction_type,
        )?;
        let variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).max(1e-12)
            * (1.0 - alpha_bar_t / alpha_bar_prev.max(1e-12));
        let stddev = self.cfg.eta * variance.max(0.0).sqrt();
        let coef_x0 = alpha_bar_prev.max(0.0).sqrt();
        let coef_eps = (1.0 - alpha_bar_prev - stddev * stddev).max(0.0).sqrt();
        let prev_mean = x0
            .affine(coef_x0, 0.0)?
            .add(&eps.affine(coef_eps, 0.0)?)?;
        let prev_sample = if stddev > 0.0 {
            prev_mean.add(&noise.affine(stddev, 0.0)?)?
        } else {
            prev_mean
        };
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
        self.rng = Philox4x32::new(self.cfg.seed);
    }
}

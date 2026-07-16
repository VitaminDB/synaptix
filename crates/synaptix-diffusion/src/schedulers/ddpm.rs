use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, randn_like, timesteps_from_spacing, BetaConfig,
    PredictionType, Scheduler, SchedulerOutput, TimestepSpacing,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarianceType {
    FixedSmall,
    FixedSmallLog,
    FixedLarge,
    FixedLargeLog,
    Learned,
    LearnedRange,
}

#[derive(Debug, Clone)]
pub struct DdpmConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub variance_type: VarianceType,
    pub spacing: TimestepSpacing,
    pub clip_sample: Option<f32>,
    pub thresholding: bool,
    pub dynamic_threshold_ratio: f32,
    pub dynamic_threshold_max: f32,
    pub seed: u64,
}

impl Default for DdpmConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            variance_type: VarianceType::FixedSmall,
            spacing: TimestepSpacing::Leading,
            clip_sample: Some(1.0),
            thresholding: false,
            dynamic_threshold_ratio: 0.995,
            dynamic_threshold_max: 1.0,
            seed: 0,
        }
    }
}

pub struct DdpmScheduler {
    cfg: DdpmConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    rng: Philox4x32,
}

impl DdpmScheduler {
    pub fn new(cfg: DdpmConfig) -> Self {
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

    fn variance(&self, t_idx: usize, t_prev: Option<usize>) -> f32 {
        let alpha_bar_t = self.alphas_cum[t_idx];
        let alpha_bar_prev = match t_prev {
            Some(i) => self.alphas_cum[i],
            None => 1.0,
        };
        let beta_t = 1.0 - alpha_bar_t / alpha_bar_prev.max(1e-12);
        let post_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).max(1e-12);
        match self.cfg.variance_type {
            VarianceType::FixedSmall | VarianceType::FixedSmallLog => post_var.max(1e-20),
            VarianceType::FixedLarge | VarianceType::FixedLargeLog => beta_t.max(1e-20),
            VarianceType::Learned | VarianceType::LearnedRange => post_var.max(1e-20),
        }
    }
}

impl Scheduler for DdpmScheduler {
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
        let t_prev = if step_idx + 1 < self.timestep_indices.len() {
            Some(self.timestep_indices[step_idx + 1])
        } else {
            None
        };
        let alpha_bar_t = self.alphas_cum[t_idx];
        let alpha_bar_prev = match t_prev {
            Some(i) => self.alphas_cum[i],
            None => 1.0,
        };
        let beta_t = 1.0 - alpha_bar_t / alpha_bar_prev.max(1e-12);
        let alpha_t = (1.0 - beta_t).max(1e-12);

        let (mut x0, _eps) = crate::schedulers::ddim::ddpm_x0_and_eps(
            sample,
            model_output,
            alpha_bar_t,
            self.cfg.prediction_type,
        )?;
        x0 = clip_or_threshold(&x0, &self.cfg)?;

        let coef_x0 = alpha_bar_prev.max(1e-12).sqrt() * beta_t / (1.0 - alpha_bar_t).max(1e-12);
        let coef_x = alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).max(1e-12);
        let mean = x0.affine(coef_x0, 0.0)?.add(&sample.affine(coef_x, 0.0)?)?;

        let prev_sample = if t_prev.is_some() {
            let var = self.variance(t_idx, t_prev);
            let std = var.sqrt();
            let noise = randn_like(&mean, &mut self.rng)?;
            mean.add(&noise.affine(std, 0.0)?)?
        } else {
            mean
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
        let t_prev = if step_idx + 1 < self.timestep_indices.len() {
            Some(self.timestep_indices[step_idx + 1])
        } else {
            None
        };
        let alpha_bar_t = self.alphas_cum[t_idx];
        let alpha_bar_prev = match t_prev {
            Some(i) => self.alphas_cum[i],
            None => 1.0,
        };
        let beta_t = 1.0 - alpha_bar_t / alpha_bar_prev.max(1e-12);
        let alpha_t = (1.0 - beta_t).max(1e-12);
        let (mut x0, _eps) = crate::schedulers::ddim::ddpm_x0_and_eps(
            sample,
            model_output,
            alpha_bar_t,
            self.cfg.prediction_type,
        )?;
        x0 = clip_or_threshold(&x0, &self.cfg)?;

        let coef_x0 = alpha_bar_prev.max(1e-12).sqrt() * beta_t / (1.0 - alpha_bar_t).max(1e-12);
        let coef_x = alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t).max(1e-12);
        let mean = x0.affine(coef_x0, 0.0)?.add(&sample.affine(coef_x, 0.0)?)?;
        let prev_sample = if t_prev.is_some() {
            let var = self.variance(t_idx, t_prev);
            let std = var.sqrt();
            mean.add(&noise.affine(std, 0.0)?)?
        } else {
            mean
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

fn clip_or_threshold(x0: &Tensor, cfg: &DdpmConfig) -> Result<Tensor> {
    if cfg.thresholding {
        let dims = x0.dims().to_vec();
        let numel = x0.numel();
        let batch = *dims.first().unwrap_or(&1);
        let per_batch = if batch > 0 { numel / batch } else { numel };
        let flat = x0.reshape((batch, per_batch))?;
        let data: Vec<Vec<f32>> = flat.to_vec2()?;
        let mut out = Vec::with_capacity(numel);
        for row in &data {
            let mut abs: Vec<f32> = row.iter().map(|v| v.abs()).collect();
            abs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let k = ((abs.len() as f32 - 1.0) * cfg.dynamic_threshold_ratio).round() as usize;
            let s = abs[k.min(abs.len() - 1)].max(1.0).min(cfg.dynamic_threshold_max);
            for &v in row {
                out.push((v / s).clamp(-1.0, 1.0) * s);
            }
        }
        return Tensor::from_vec(out, dims, x0.device()).map_err(DiffusionError::from);
    }
    if let Some(c) = cfg.clip_sample {
        return x0.clamp(-c, c).map_err(DiffusionError::from);
    }
    Ok(x0.clone())
}

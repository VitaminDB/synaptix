use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, alphas_to_sigmas, betas_for, convert_to_x0, exponential_sigmas, karras_sigmas,
    randn_like, timesteps_from_spacing, BetaConfig, PredictionType, Scheduler, SchedulerOutput,
    TimestepSpacing,
};
use crate::schedulers::euler::SigmaSchedule;

#[derive(Debug, Clone)]
pub struct EulerAncestralConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub sigma_schedule: SigmaSchedule,
    pub karras_rho: f32,
    pub eta: f32,
    pub use_scale_model_input: bool,
    pub seed: u64,
}

impl Default for EulerAncestralConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Leading,
            sigma_schedule: SigmaSchedule::BetaSchedule,
            karras_rho: 7.0,
            eta: 1.0,
            use_scale_model_input: true,
            seed: 0,
        }
    }
}

pub struct EulerAncestralScheduler {
    cfg: EulerAncestralConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    rng: Philox4x32,
}

impl EulerAncestralScheduler {
    pub fn new(cfg: EulerAncestralConfig) -> Self {
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
}

fn ancestral_split(sigma_cur: f32, sigma_next: f32, eta: f32) -> (f32, f32) {
    let s_up = eta
        * ((sigma_next * sigma_next * (sigma_cur * sigma_cur - sigma_next * sigma_next))
            / sigma_cur.max(1e-12).powi(2))
        .max(0.0)
        .sqrt();
    let s_down = (sigma_next * sigma_next - s_up * s_up).max(0.0).sqrt();
    (s_down, s_up)
}

impl Scheduler for EulerAncestralScheduler {
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
                self.sigmas = karras_sigmas(*all.first().unwrap(), *all.last().unwrap(), n_steps, self.cfg.karras_rho);
            }
            SigmaSchedule::Exponential => {
                let all = alphas_to_sigmas(&self.alphas_cum);
                self.sigmas = exponential_sigmas(*all.first().unwrap(), *all.last().unwrap(), n_steps);
            }
        }
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
        let (s_down, s_up) = ancestral_split(sigma, sigma_next, self.cfg.eta);
        let x0 = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;
        let d = sample.sub(&x0)?.affine(1.0 / sigma.max(1e-12), 0.0)?;
        let dt = s_down - sigma;
        let mut prev = sample.add(&d.affine(dt, 0.0)?)?;
        if s_up > 0.0 {
            let noise = randn_like(&prev, &mut self.rng)?;
            prev = prev.add(&noise.affine(s_up, 0.0)?)?;
        }
        Ok(SchedulerOutput {
            prev_sample: prev,
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
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange {
                idx: step_idx,
                n_steps: self.sigmas.len().saturating_sub(1),
            });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let (s_down, s_up) = ancestral_split(sigma, sigma_next, self.cfg.eta);
        let x0 = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;
        let d = sample.sub(&x0)?.affine(1.0 / sigma.max(1e-12), 0.0)?;
        let dt = s_down - sigma;
        let mut prev = sample.add(&d.affine(dt, 0.0)?)?;
        if s_up > 0.0 {
            prev = prev.add(&noise.affine(s_up, 0.0)?)?;
        }
        Ok(SchedulerOutput {
            prev_sample: prev,
            pred_original_sample: Some(x0),
        })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        crate::schedulers::add_noise_ve(original, noise, sigma)
    }

    fn reset_state(&mut self) {
        self.rng = Philox4x32::new(self.cfg.seed);
    }
}

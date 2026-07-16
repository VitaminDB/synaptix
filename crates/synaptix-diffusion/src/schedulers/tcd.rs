use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, convert_to_x0, randn_like, BetaConfig, PredictionType, Scheduler,
    SchedulerOutput,
};

#[derive(Debug, Clone)]
pub struct TcdConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub original_inference_steps: usize,
    pub gamma: f32,
    pub clip_denoised: Option<f32>,
    pub seed: u64,
}

impl Default for TcdConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Velocity,
            original_inference_steps: 50,
            gamma: 0.3,
            clip_denoised: Some(1.0),
            seed: 0,
        }
    }
}

pub struct TcdScheduler {
    cfg: TcdConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    rng: Philox4x32,
}

impl TcdScheduler {
    pub fn new(cfg: TcdConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        let rng = Philox4x32::new(cfg.seed);
        Self { cfg, alphas_cum, sigmas: Vec::new(), timesteps: Vec::new(), timestep_indices: Vec::new(), rng }
    }
}

impl Scheduler for TcdScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        let step = n_train / self.cfg.original_inference_steps.max(1);
        let k = step;
        let c = self.cfg.original_inference_steps / n_steps.max(1);
        let mut indices: Vec<usize> = (0..n_steps)
            .map(|i| ((n_steps - 1 - i) * c * k).min(n_train - 1))
            .collect();
        indices.sort_by(|a, b| b.cmp(a));
        indices.dedup();
        self.timestep_indices = indices;
        self.timesteps = self.timestep_indices.iter().map(|&i| i as f32).collect();
        let mut sigmas: Vec<f32> = self.timestep_indices.iter()
            .map(|&i| { let a = self.alphas_cum[i].clamp(1e-12, 1.0 - 1e-12); ((1.0 - a) / a).sqrt() })
            .collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { self.cfg.prediction_type }

    fn step(&mut self, model_output: &Tensor, step_idx: usize, sample: &Tensor) -> Result<SchedulerOutput> {
        if step_idx >= self.timestep_indices.len() {
            return Err(DiffusionError::StepOutOfRange { idx: step_idx, n_steps: self.timestep_indices.len() });
        }
        let t_idx = self.timestep_indices[step_idx];
        let t_s = if step_idx + 1 < self.timestep_indices.len() {
            let next = self.timestep_indices[step_idx + 1];
            let gap = (t_idx - next).max(0);
            let s_offset = (self.cfg.gamma * gap as f32) as usize;
            t_idx.saturating_sub(s_offset).max(next)
        } else {
            0
        };
        let alpha_bar_t = self.alphas_cum[t_idx];
        let alpha_bar_s = self.alphas_cum[t_s.min(self.alphas_cum.len() - 1)];
        let sigma = ((1.0 - alpha_bar_t) / alpha_bar_t.max(1e-12)).sqrt();
        let x0 = convert_to_x0(sample, model_output, sigma, self.cfg.prediction_type)?;
        let x0 = if let Some(c) = self.cfg.clip_denoised { x0.clamp(-c, c)? } else { x0 };
        let noise = randn_like(&x0, &mut self.rng)?;
        let prev_sample = x0.affine(alpha_bar_s.max(1e-12).sqrt(), 0.0)?
            .add(&noise.affine((1.0 - alpha_bar_s).max(0.0).sqrt(), 0.0)?)?;
        Ok(SchedulerOutput { prev_sample, pred_original_sample: Some(x0) })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let idx = step_idx.min(self.timestep_indices.len().saturating_sub(1));
        let alpha_bar = self.alphas_cum[self.timestep_indices[idx]];
        original.affine(alpha_bar.max(1e-12).sqrt(), 0.0)?
            .add(&noise.affine((1.0 - alpha_bar).max(0.0).sqrt(), 0.0)?)
            .map_err(DiffusionError::from)
    }

    fn reset_state(&mut self) { self.rng = Philox4x32::new(self.cfg.seed); }
}

use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    convert_to_x0, karras_sigmas, randn_like, PredictionType, Scheduler, SchedulerOutput,
};

#[derive(Debug, Clone)]
pub struct EdmConfig {
    pub sigma_min: f32,
    pub sigma_max: f32,
    pub sigma_data: f32,
    pub rho: f32,
    pub s_churn: f32,
    pub s_tmin: f32,
    pub s_tmax: f32,
    pub s_noise: f32,
    pub prediction_type: PredictionType,
    pub seed: u64,
}

impl Default for EdmConfig {
    fn default() -> Self {
        Self {
            sigma_min: 0.002,
            sigma_max: 80.0,
            sigma_data: 0.5,
            rho: 7.0,
            s_churn: 0.0,
            s_tmin: 0.0,
            s_tmax: f32::INFINITY,
            s_noise: 1.0,
            prediction_type: PredictionType::Epsilon,
            seed: 0,
        }
    }
}

pub struct EdmScheduler {
    cfg: EdmConfig,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    rng: Philox4x32,
}

impl EdmScheduler {
    pub fn new(cfg: EdmConfig) -> Self {
        let rng = Philox4x32::new(cfg.seed);
        Self { cfg, sigmas: Vec::new(), timesteps: Vec::new(), rng }
    }

    pub fn c_skip(&self, sigma: f32) -> f32 {
        let sd = self.cfg.sigma_data;
        sd * sd / (sigma * sigma + sd * sd)
    }

    pub fn c_out(&self, sigma: f32) -> f32 {
        let sd = self.cfg.sigma_data;
        sigma * sd / (sigma * sigma + sd * sd).sqrt()
    }

    fn c_in(&self, sigma: f32) -> f32 {
        1.0 / (self.cfg.sigma_data * self.cfg.sigma_data + sigma * sigma).sqrt()
    }
}

impl Scheduler for EdmScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        self.sigmas = karras_sigmas(self.cfg.sigma_min, self.cfg.sigma_max, n_steps, self.cfg.rho);
        self.timesteps = self.sigmas[..n_steps].to_vec();
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { self.cfg.prediction_type }

    fn scale_model_input(&self, sample: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(1.0);
        let c_in = self.c_in(sigma);
        sample.affine(c_in, 0.0).map_err(DiffusionError::from)
    }

    fn init_noise_sigma(&self) -> f32 {
        (self.cfg.sigma_max * self.cfg.sigma_max + self.cfg.sigma_data * self.cfg.sigma_data).sqrt()
    }

    fn step(&mut self, model_output: &Tensor, step_idx: usize, sample: &Tensor) -> Result<SchedulerOutput> {
        if step_idx + 1 >= self.sigmas.len() {
            return Err(DiffusionError::StepOutOfRange { idx: step_idx, n_steps: self.sigmas.len().saturating_sub(1) });
        }
        let sigma = self.sigmas[step_idx];
        let sigma_next = self.sigmas[step_idx + 1];
        let gamma = if self.cfg.s_churn > 0.0 && sigma >= self.cfg.s_tmin && sigma <= self.cfg.s_tmax {
            let g = (self.cfg.s_churn / self.sigmas.len() as f32).min(2_f32.sqrt() - 1.0);
            g.min(sigma_next / sigma - 1.0).max(0.0)
        } else {
            0.0
        };
        let sigma_hat = sigma * (1.0 + gamma);
        let cur_sample = if gamma > 0.0 {
            let eps = randn_like(sample, &mut self.rng)?;
            sample.add(&eps.affine((sigma_hat * sigma_hat - sigma * sigma).max(0.0).sqrt() * self.cfg.s_noise, 0.0)?)?
        } else {
            sample.clone()
        };
        let x0 = convert_to_x0(&cur_sample, model_output, sigma_hat, self.cfg.prediction_type)?;
        let d = cur_sample.sub(&x0)?.affine(1.0 / sigma_hat.max(1e-12), 0.0)?;
        let dt = sigma_next - sigma_hat;
        let prev_sample = cur_sample.add(&d.affine(dt, 0.0)?)?;
        Ok(SchedulerOutput { prev_sample, pred_original_sample: Some(x0) })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        crate::schedulers::add_noise_ve(original, noise, sigma)
    }

    fn reset_state(&mut self) { self.rng = Philox4x32::new(self.cfg.seed); }
}

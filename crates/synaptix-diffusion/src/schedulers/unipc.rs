use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, convert_to_x0, timesteps_from_spacing, BetaConfig, PredictionType,
    Scheduler, SchedulerOutput, TimestepSpacing,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UniPcVariant {
    Bh1,
    Bh2,
}

#[derive(Debug, Clone)]
pub struct UniPcConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub solver_order: usize,
    pub variant: UniPcVariant,
    pub thresholding: bool,
    pub dynamic_threshold_ratio: f32,
}

impl Default for UniPcConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig::default(),
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Linspace,
            solver_order: 2,
            variant: UniPcVariant::Bh2,
            thresholding: false,
            dynamic_threshold_ratio: 0.995,
        }
    }
}

pub struct UniPcScheduler {
    cfg: UniPcConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    timestep_indices: Vec<usize>,
    model_outputs: Vec<Tensor>,
    timestep_list: Vec<usize>,
}

impl UniPcScheduler {
    pub fn new(cfg: UniPcConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            timestep_indices: Vec::new(),
            model_outputs: Vec::new(),
            timestep_list: Vec::new(),
        }
    }

    fn lambda(&self, t_idx: usize) -> f32 {
        let a = self.alphas_cum[t_idx].clamp(1e-12, 1.0 - 1e-12);
        let alpha = a.sqrt();
        let sigma = (1.0 - a).sqrt();
        alpha.ln() - sigma.ln()
    }
}

impl Scheduler for UniPcScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        self.timestep_indices = timesteps_from_spacing(n_train, n_steps + 1, self.cfg.spacing, 1);
        self.timestep_indices.truncate(n_steps);
        self.timesteps = self.timestep_indices.iter().map(|&i| i as f32).collect();
        let mut sigmas: Vec<f32> = self.timestep_indices.iter()
            .map(|&i| { let a = self.alphas_cum[i].clamp(1e-12, 1.0 - 1e-12); ((1.0 - a) / a).sqrt() })
            .collect();
        sigmas.push(0.0);
        self.sigmas = sigmas;
        self.model_outputs.clear();
        self.timestep_list.clear();
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { &self.timesteps }
    fn sigmas(&self) -> &[f32] { &self.sigmas }
    fn prediction_type(&self) -> PredictionType { self.cfg.prediction_type }

    fn step(&mut self, model_output: &Tensor, step_idx: usize, sample: &Tensor) -> Result<SchedulerOutput> {
        if step_idx >= self.timestep_indices.len() {
            return Err(DiffusionError::StepOutOfRange { idx: step_idx, n_steps: self.timestep_indices.len() });
        }
        let t_cur = self.timestep_indices[step_idx];
        let t_prev = if step_idx + 1 < self.timestep_indices.len() {
            self.timestep_indices[step_idx + 1]
        } else {
            0
        };
        let alpha_bar_t = self.alphas_cum[t_cur];
        let alpha_t = alpha_bar_t.max(1e-12).sqrt();
        let sigma_t = (1.0 - alpha_bar_t).max(0.0).sqrt();
        let lambda_t = self.lambda(t_cur);
        let lambda_s = self.lambda(t_prev.min(self.alphas_cum.len() - 1));
        let h = lambda_s - lambda_t;
        let a_bar_prev = self.alphas_cum[t_prev.min(self.alphas_cum.len() - 1)];
        let alpha_prev = a_bar_prev.max(1e-12).sqrt();
        let sigma_prev = (1.0 - a_bar_prev).max(0.0).sqrt();

        let sigma_cur = ((1.0 - alpha_bar_t) / alpha_bar_t.max(1e-12)).sqrt();
        let x0 = convert_to_x0(sample, model_output, sigma_cur, self.cfg.prediction_type)?;

        let phi1 = (h.exp() - 1.0) / h;
        let rho_x0 = alpha_prev / alpha_t;
        let _rho_n = sigma_prev * h.exp() - sigma_t * phi1;

        let prev_sample = if self.model_outputs.is_empty()
            || self.cfg.solver_order < 2
            || matches!(self.cfg.variant, UniPcVariant::Bh1)
        {
            sample.affine(rho_x0, 0.0)?.sub(&sample.affine(sigma_t / alpha_t, 0.0)?)?
                .add(&x0.affine(alpha_prev + sigma_prev * phi1, 0.0)?)?
        } else {
            let prev_out = &self.model_outputs[self.model_outputs.len() - 1];
            let t_prev2 = self.timestep_list[self.timestep_list.len() - 1];
            let lambda_t2 = self.lambda(t_prev2);
            let h_last = lambda_t - lambda_t2;
            let r = h_last / h;
            let a_prev2 = self.alphas_cum[t_prev2.min(self.alphas_cum.len() - 1)];
            let alpha_prev2 = a_prev2.max(1e-12).sqrt();
            let sigma_prev2 = (1.0 - a_prev2).max(0.0).sqrt();
            let phi2 = (h.exp() - 1.0 - h) / (h * h);
            let d0 = x0.affine(1.0 / (sigma_t + alpha_t), 0.0)?;
            let d1_t = convert_to_x0(sample, prev_out, ((1.0 - a_prev2) / a_prev2.max(1e-12)).sqrt(), self.cfg.prediction_type)?;
            let d1 = d0.sub(&d1_t.affine(1.0 / (sigma_prev2 + alpha_prev2), 0.0)?)?.affine(1.0 / r, 0.0)?;
            let coef = alpha_prev + sigma_prev * phi1;
            let coef2 = sigma_prev * phi2 * h;
            sample.affine(rho_x0, 0.0)?
                .add(&x0.affine(coef, 0.0)?)?
                .add(&d1.affine(coef2, 0.0)?)?
        };

        if self.model_outputs.len() == self.cfg.solver_order {
            self.model_outputs.remove(0);
            self.timestep_list.remove(0);
        }
        self.model_outputs.push(model_output.clone());
        self.timestep_list.push(t_cur);

        Ok(SchedulerOutput { prev_sample, pred_original_sample: Some(x0) })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let idx = step_idx.min(self.timestep_indices.len().saturating_sub(1));
        let alpha_bar = self.alphas_cum[self.timestep_indices[idx]];
        original.affine(alpha_bar.max(1e-12).sqrt(), 0.0)?
            .add(&noise.affine((1.0 - alpha_bar).max(0.0).sqrt(), 0.0)?)
            .map_err(DiffusionError::from)
    }

    fn reset_state(&mut self) {
        self.model_outputs.clear();
        self.timestep_list.clear();
    }
}

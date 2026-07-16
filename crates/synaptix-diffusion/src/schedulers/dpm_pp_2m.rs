use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{
    alphas_cumprod, betas_for, BetaConfig, PredictionType, Scheduler, SchedulerOutput,
    TimestepSpacing,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlgorithmType {
    DpmSolver,
    DpmSolverPlusPlus,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverType {
    Midpoint,
    Heun,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinalSigmasType {
    Zero,
    SigmaMin,
}

#[derive(Debug, Clone)]
pub struct DpmPp2MConfig {
    pub beta: BetaConfig,
    pub prediction_type: PredictionType,
    pub spacing: TimestepSpacing,
    pub algorithm_type: AlgorithmType,
    pub solver_type: SolverType,
    pub lower_order_final: bool,
    pub final_sigmas_type: FinalSigmasType,
    pub solver_order: usize,
}

impl Default for DpmPp2MConfig {
    fn default() -> Self {
        Self {
            beta: BetaConfig {
                num_train_timesteps: 1000,
                beta_start: 0.0001,
                beta_end: 0.02,
                schedule: crate::schedulers::BetaSchedule::Linear,
                rescale_zero_snr: false,
            },
            prediction_type: PredictionType::Epsilon,
            spacing: TimestepSpacing::Linspace,
            algorithm_type: AlgorithmType::DpmSolverPlusPlus,
            solver_type: SolverType::Midpoint,
            lower_order_final: true,
            final_sigmas_type: FinalSigmasType::Zero,
            solver_order: 2,
        }
    }
}

pub struct DpmPp2MScheduler {
    cfg: DpmPp2MConfig,
    alphas_cum: Vec<f32>,
    sigmas: Vec<f32>,
    timesteps: Vec<f32>,
    n_steps: usize,
    step_index: usize,
    model_outputs: Vec<Option<Tensor>>,
    lower_order_nums: usize,
}

impl DpmPp2MScheduler {
    pub fn new(cfg: DpmPp2MConfig) -> Self {
        let betas = betas_for(&cfg.beta);
        let alphas_cum = alphas_cumprod(&betas);
        let order = cfg.solver_order;
        Self {
            cfg,
            alphas_cum,
            sigmas: Vec::new(),
            timesteps: Vec::new(),
            n_steps: 0,
            step_index: 0,
            model_outputs: vec![None; order],
            lower_order_nums: 0,
        }
    }

    fn compute_timesteps_dpm(&self, n_steps: usize) -> Vec<f32> {
        let n_train = self.cfg.beta.num_train_timesteps;
        let last = (n_train - 1) as f32;
        let total = (n_steps + 1) as f32;
        let raw: Vec<f32> = (0..=n_steps)
            .map(|i| (last * (i as f32) / (total - 1.0)).round())
            .collect();
        raw.iter().rev().take(n_steps).copied().collect()
    }

    fn interp_sigmas(&self, timesteps_float: &[f32]) -> Vec<f32> {
        let sigmas_full: Vec<f32> = self
            .alphas_cum
            .iter()
            .map(|&a| {
                let a = a.clamp(1e-12, 1.0 - 1e-12);
                ((1.0 - a) / a).sqrt()
            })
            .collect();
        let n_full = sigmas_full.len();
        timesteps_float
            .iter()
            .map(|&t| {
                let low = (t.floor() as i64).clamp(0, (n_full - 1) as i64) as usize;
                let high = (low + 1).min(n_full - 1);
                let frac = t - low as f32;
                sigmas_full[low] * (1.0 - frac) + sigmas_full[high] * frac
            })
            .collect()
    }

    fn sigma_to_alpha_sigma_t(sigma: f32) -> (f32, f32) {
        let alpha = 1.0 / (sigma * sigma + 1.0).sqrt();
        let sig = sigma * alpha;
        (alpha, sig)
    }

    fn convert_to_x0(&self, sample: &Tensor, model_output: &Tensor, sigma: f32) -> Result<Tensor> {
        let (alpha, sigma_t) = Self::sigma_to_alpha_sigma_t(sigma);
        match self.cfg.prediction_type {
            PredictionType::Epsilon => sample
                .sub(&model_output.affine(sigma_t, 0.0)?)?
                .affine(1.0 / alpha, 0.0)
                .map_err(DiffusionError::from),
            PredictionType::SampleX0 => Ok(model_output.clone()),
            PredictionType::Velocity => sample
                .affine(alpha, 0.0)?
                .sub(&model_output.affine(sigma_t, 0.0)?)
                .map_err(DiffusionError::from),
            PredictionType::FlowMatchVelocity => Err(DiffusionError::invalid_arg(
                "DPM++: FlowMatchVelocity не поддерживается",
            )),
        }
    }

    fn first_order_update(&self, x0_pred: &Tensor, sample: &Tensor) -> Result<Tensor> {
        let sigma_s = self.sigmas[self.step_index];
        let sigma_t_raw = self.sigmas[self.step_index + 1];
        let (alpha_s, sigma_s_new) = Self::sigma_to_alpha_sigma_t(sigma_s);
        let (alpha_t, sigma_t_new) = Self::sigma_to_alpha_sigma_t(sigma_t_raw);
        let lambda_s = alpha_s.ln() - sigma_s_new.ln();
        let lambda_t = alpha_t.ln() - sigma_t_new.ln();
        let h = lambda_t - lambda_s;
        let coef_sample = sigma_t_new / sigma_s_new;
        let coef_x0 = -alpha_t * ((-h).exp() - 1.0);
        sample
            .affine(coef_sample, 0.0)?
            .add(&x0_pred.affine(coef_x0, 0.0)?)
            .map_err(DiffusionError::from)
    }

    fn second_order_update(&self, sample: &Tensor) -> Result<Tensor> {
        let sigma_t_raw = self.sigmas[self.step_index + 1];
        let sigma_s0_raw = self.sigmas[self.step_index];
        let sigma_s1_raw = self.sigmas[self.step_index - 1];
        let (alpha_t, sigma_t_new) = Self::sigma_to_alpha_sigma_t(sigma_t_raw);
        let (alpha_s0, sigma_s0_new) = Self::sigma_to_alpha_sigma_t(sigma_s0_raw);
        let (alpha_s1, sigma_s1_new) = Self::sigma_to_alpha_sigma_t(sigma_s1_raw);
        let lambda_t = alpha_t.ln() - sigma_t_new.ln();
        let lambda_s0 = alpha_s0.ln() - sigma_s0_new.ln();
        let lambda_s1 = alpha_s1.ln() - sigma_s1_new.ln();
        let h = lambda_t - lambda_s0;
        let h_0 = lambda_s0 - lambda_s1;
        let r0 = h_0 / h;

        let m0 = self.model_outputs[self.cfg.solver_order - 1]
            .as_ref()
            .ok_or_else(|| DiffusionError::invalid_arg("DPM++: m0 не задано"))?;
        let m1 = self.model_outputs[self.cfg.solver_order - 2]
            .as_ref()
            .ok_or_else(|| DiffusionError::invalid_arg("DPM++: m1 не задано"))?;

        let d0 = m0.clone();
        let d1 = m0.sub(m1)?.affine(1.0 / r0, 0.0)?;

        let coef_sample = sigma_t_new / sigma_s0_new;
        let coef_d0 = -alpha_t * ((-h).exp() - 1.0);
        let coef_d1 = match self.cfg.solver_type {
            SolverType::Midpoint => -0.5 * alpha_t * ((-h).exp() - 1.0),
            SolverType::Heun => alpha_t * (((-h).exp() - 1.0) / h + 1.0),
        };
        sample
            .affine(coef_sample, 0.0)?
            .add(&d0.affine(coef_d0, 0.0)?)?
            .add(&d1.affine(coef_d1, 0.0)?)
            .map_err(DiffusionError::from)
    }

    pub fn n_steps(&self) -> usize {
        self.n_steps
    }
}

impl Scheduler for DpmPp2MScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        if n_steps == 0 {
            return Err(DiffusionError::invalid_arg("n_steps must be > 0"));
        }
        let n_train = self.cfg.beta.num_train_timesteps;
        if n_steps >= n_train {
            return Err(DiffusionError::invalid_arg(format!(
                "n_steps={n_steps} >= num_train_timesteps={n_train}"
            )));
        }
        let timesteps = self.compute_timesteps_dpm(n_steps);
        let mut sigmas = self.interp_sigmas(&timesteps);
        let sigma_last = match self.cfg.final_sigmas_type {
            FinalSigmasType::Zero => 0.0,
            FinalSigmasType::SigmaMin => {
                let a = self.alphas_cum[0].clamp(1e-12, 1.0 - 1e-12);
                ((1.0 - a) / a).sqrt()
            }
        };
        sigmas.push(sigma_last);
        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.n_steps = n_steps;
        self.step_index = 0;
        self.model_outputs = vec![None; self.cfg.solver_order];
        self.lower_order_nums = 0;
        Ok(())
    }

    fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas.first().copied().unwrap_or(1.0)
    }

    fn prediction_type(&self) -> PredictionType {
        self.cfg.prediction_type
    }

    fn scale_model_input(&self, sample: &Tensor, step_idx: usize) -> Result<Tensor> {
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
        if step_idx != self.step_index {
            return Err(DiffusionError::invalid_arg(format!(
                "DPM++: step_idx={step_idx} не совпадает с internal step_index={}",
                self.step_index
            )));
        }
        let sigma = self.sigmas[step_idx];
        let x0_pred = self.convert_to_x0(sample, model_output, sigma)?;

        for i in 0..(self.cfg.solver_order - 1) {
            self.model_outputs[i] = self.model_outputs[i + 1].take();
        }
        self.model_outputs[self.cfg.solver_order - 1] = Some(x0_pred.clone());

        let is_last_step = step_idx == self.n_steps - 1;
        let lower_order_final = self.cfg.lower_order_final && is_last_step;
        let need_first_order =
            step_idx == 0 || lower_order_final || self.lower_order_nums < 1;

        let prev_sample = if need_first_order {
            self.first_order_update(&x0_pred, sample)?
        } else {
            self.second_order_update(sample)?
        };

        if self.lower_order_nums < self.cfg.solver_order {
            self.lower_order_nums += 1;
        }
        self.step_index += 1;

        Ok(SchedulerOutput {
            prev_sample,
            pred_original_sample: Some(x0_pred),
        })
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor> {
        let sigma = self.sigmas.get(step_idx).copied().unwrap_or(0.0);
        let (alpha, sigma_t) = Self::sigma_to_alpha_sigma_t(sigma);
        original
            .affine(alpha, 0.0)?
            .add(&noise.affine(sigma_t, 0.0)?)
            .map_err(DiffusionError::from)
    }

    fn reset_state(&mut self) {
        self.step_index = 0;
        self.model_outputs = vec![None; self.cfg.solver_order];
        self.lower_order_nums = 0;
    }
}

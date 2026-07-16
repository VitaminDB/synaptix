use crate::schedulers::euler::{EulerConfig, EulerScheduler, SigmaSchedule};
use crate::schedulers::{BetaConfig, Scheduler, TimestepSpacing};

pub const SDXL_TURBO_4STEP: [f32; 5] = [14.615, 6.475, 3.861, 1.886, 0.0];
pub const SD1_TURBO_4STEP: [f32; 5] = [14.615, 7.977, 3.861, 1.886, 0.0];
pub const LIGHTNING_4STEP: [f32; 5] = [14.615, 6.475, 2.697, 0.963, 0.0];
pub const LIGHTNING_8STEP: [f32; 9] = [14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.652, 0.399, 0.0];

#[derive(Debug, Clone)]
pub struct DistilledConfig {
    pub beta: BetaConfig,
    pub sigmas: Vec<f32>,
}

impl DistilledConfig {
    pub fn sdxl_turbo_4step() -> Self {
        Self { beta: BetaConfig::default(), sigmas: SDXL_TURBO_4STEP.to_vec() }
    }

    pub fn lightning_4step() -> Self {
        Self { beta: BetaConfig::default(), sigmas: LIGHTNING_4STEP.to_vec() }
    }

    pub fn lightning_8step() -> Self {
        Self { beta: BetaConfig::default(), sigmas: LIGHTNING_8STEP.to_vec() }
    }
}

pub struct DistilledScheduler {
    inner: EulerScheduler,
    fixed_sigmas: Vec<f32>,
}

impl DistilledScheduler {
    pub fn new(cfg: DistilledConfig) -> Self {
        let euler_cfg = EulerConfig {
            beta: cfg.beta,
            sigma_schedule: SigmaSchedule::BetaSchedule,
            spacing: TimestepSpacing::Leading,
            ..Default::default()
        };
        let inner = EulerScheduler::new(euler_cfg);
        Self { inner, fixed_sigmas: cfg.sigmas }
    }
}

impl Scheduler for DistilledScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> crate::error::Result<()> {
        self.inner.set_timesteps(n_steps)?;
        let want = n_steps + 1;
        if self.fixed_sigmas.len() >= want {
            let s = self.fixed_sigmas[..want].to_vec();
            self.inner.override_sigmas(s);
        }
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { self.inner.timesteps() }
    fn sigmas(&self) -> &[f32] { self.inner.sigmas() }
    fn prediction_type(&self) -> crate::schedulers::PredictionType { self.inner.prediction_type() }

    fn scale_model_input(&self, sample: &synaptix_core::tensor::Tensor, step_idx: usize) -> crate::error::Result<synaptix_core::tensor::Tensor> {
        self.inner.scale_model_input(sample, step_idx)
    }

    fn step(&mut self, model_output: &synaptix_core::tensor::Tensor, step_idx: usize, sample: &synaptix_core::tensor::Tensor) -> crate::error::Result<crate::schedulers::SchedulerOutput> {
        self.inner.step(model_output, step_idx, sample)
    }

    fn add_noise(&self, original: &synaptix_core::tensor::Tensor, noise: &synaptix_core::tensor::Tensor, step_idx: usize) -> crate::error::Result<synaptix_core::tensor::Tensor> {
        self.inner.add_noise(original, noise, step_idx)
    }
}

use crate::error::Result;
use crate::schedulers::euler::{EulerConfig, EulerScheduler, SigmaSchedule};
use crate::schedulers::{BetaConfig, Scheduler, TimestepSpacing};

pub const AYS_11_SD1: [f32; 11] = [
    14.615, 6.475, 3.861, 2.697, 1.886, 1.396, 0.963, 0.652, 0.399, 0.152, 0.029,
];
pub const AYS_11_SDXL: [f32; 11] = [
    14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029,
];
pub const AYS_11_SVD: [f32; 11] = [
    700.00, 54.5, 15.886, 7.977, 4.248, 1.789, 0.981, 0.403, 0.173, 0.034, 0.002,
];

#[derive(Debug, Clone)]
pub struct AysConfig {
    pub beta: BetaConfig,
    pub sigmas: Vec<f32>,
}

impl AysConfig {
    pub fn sdxl() -> Self {
        Self { beta: BetaConfig::default(), sigmas: AYS_11_SDXL.to_vec() }
    }

    pub fn sd1() -> Self {
        Self { beta: BetaConfig::default(), sigmas: AYS_11_SD1.to_vec() }
    }
}

pub struct AysScheduler {
    inner: EulerScheduler,
    fixed_sigmas: Vec<f32>,
}

impl AysScheduler {
    pub fn new(cfg: AysConfig) -> Self {
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

impl Scheduler for AysScheduler {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()> {
        self.inner.set_timesteps(n_steps)?;
        if self.fixed_sigmas.len() >= n_steps {
            let end = n_steps + 1;
            let mut s = self.fixed_sigmas[..end.min(self.fixed_sigmas.len())].to_vec();
            if s.last() != Some(&0.0) {
                s.push(0.0);
            }
            self.inner.override_sigmas(s);
        }
        Ok(())
    }

    fn timesteps(&self) -> &[f32] { self.inner.timesteps() }
    fn sigmas(&self) -> &[f32] { self.inner.sigmas() }
    fn prediction_type(&self) -> crate::schedulers::PredictionType { self.inner.prediction_type() }

    fn scale_model_input(&self, sample: &synaptix_core::tensor::Tensor, step_idx: usize) -> Result<synaptix_core::tensor::Tensor> {
        self.inner.scale_model_input(sample, step_idx)
    }

    fn step(&mut self, model_output: &synaptix_core::tensor::Tensor, step_idx: usize, sample: &synaptix_core::tensor::Tensor) -> Result<crate::schedulers::SchedulerOutput> {
        self.inner.step(model_output, step_idx, sample)
    }

    fn add_noise(&self, original: &synaptix_core::tensor::Tensor, noise: &synaptix_core::tensor::Tensor, step_idx: usize) -> Result<synaptix_core::tensor::Tensor> {
        self.inner.add_noise(original, noise, step_idx)
    }
}

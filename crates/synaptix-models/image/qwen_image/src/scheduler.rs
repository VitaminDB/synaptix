//! Расписание Qwen-Image: `FlowMatchEulerDiscreteScheduler` с sigmas =
//! linspace(1, 1/N, N), экспоненциальным сдвигом по `mu` (линейно по длине
//! латента, `calculate_shift`) и растяжкой хвоста к `shift_terminal`
//! (последняя сигма перед нулём = 0.02). Шаг Эйлера в F32.

use synaptix_core::{dtype::DType, error::Result, tensor::Tensor};

use crate::config::SchedulerConfig;

pub struct QwenScheduler {
    /// Длина N+1, последний 0.
    sigmas: Vec<f32>,
}

impl QwenScheduler {
    /// Как `retrieve_timesteps(sigmas=linspace(1, 1/N, N), mu=…)` у
    /// пайплайнов: sigmas в f32, `exp(mu) / (exp(mu) + (1/σ − 1))`, затем
    /// `1 − (1 − σ)/k`, где `k` переводит последнюю сигму в `shift_terminal`.
    pub fn new(num_steps: usize, image_seq_len: usize, cfg: &SchedulerConfig) -> Self {
        let n = num_steps.max(1);
        let em = cfg.mu(image_seq_len).exp() as f32;
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                // np.linspace(1.0, 1/n, n) в f64 → astype(float32).
                let lin = if n == 1 { 1.0 } else { 1.0 + i as f64 * ((1.0 / n as f64 - 1.0) / (n - 1) as f64) } as f32;
                em / (em + (1.0 / lin - 1.0))
            })
            .collect();
        if let Some(term) = cfg.shift_terminal {
            let last = 1.0 - *sigmas.last().unwrap_or(&0.0);
            let k = last / (1.0 - term as f32);
            if k > 0.0 {
                for s in &mut sigmas {
                    *s = 1.0 - (1.0 - *s) / k;
                }
            }
        }
        sigmas.push(0.0);
        Self { sigmas }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    pub fn sigmas(&self) -> &[f32] {
        &self.sigmas
    }

    /// Шаг Эйлера: `sample + (σ[i+1] − σ[i])·v` (F32).
    pub fn step(&self, v: &Tensor, i: usize, sample: &Tensor) -> Result<Tensor> {
        let dt = self.sigmas[i + 1] - self.sigmas[i];
        sample.to_dtype(DType::F32)?.add(&v.to_dtype(DType::F32)?.mul_scalar(dt)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmas_match_reference_1024() {
        // diffusers 0.40, конфиг расписания Qwen-Image-Edit-2511, 1024² (4096
        // токенов латента), 4 шага → scheduler.sigmas.
        let s = QwenScheduler::new(4, 4096, &SchedulerConfig::default());
        let cfg = SchedulerConfig::default();
        let em = cfg.mu(4096).exp();
        let raw: Vec<f64> = [1.0, 0.75, 0.5, 0.25].iter().map(|t: &f64| em / (em + (1.0 / t - 1.0))).collect();
        let k = (1.0 - raw[3]) / 0.98;
        for (i, r) in raw.iter().enumerate() {
            let want = 1.0 - (1.0 - r) / k;
            assert!((s.sigma(i) as f64 - want).abs() < 2e-6, "{i}: {} vs {want}", s.sigma(i));
        }
        assert!((s.sigma(3) - 0.02).abs() < 1e-6);
        assert_eq!(s.sigma(4), 0.0);
        assert_eq!(s.sigma(0), 1.0);
    }
}

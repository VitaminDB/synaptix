//! Расписание FLUX.2: `FlowMatchEulerDiscreteScheduler` с sigmas =
//! linspace(1, 1/N, N), экспоненциальным сдвигом по `mu` и эмпирическим `mu`
//! из `compute_empirical_mu` пайплайна (зависит и от длины латента, и от
//! числа шагов — в отличие от FLUX.1). Шаг Эйлера в F32.

use synaptix_core::{dtype::DType, error::Result, tensor::Tensor};

/// `compute_empirical_mu` (diffusers `pipeline_flux2.py`).
pub fn empirical_mu(image_seq_len: usize, num_steps: usize) -> f64 {
    let (a1, b1) = (8.73809524e-05, 1.89833333);
    let (a2, b2) = (0.00016927, 0.45666666);
    let seq = image_seq_len as f64;
    if image_seq_len > 4300 {
        return a2 * seq + b2;
    }
    let m_200 = a2 * seq + b2;
    let m_10 = a1 * seq + b1;
    let a = (m_200 - m_10) / 190.0;
    let b = m_200 - 200.0 * a;
    a * num_steps as f64 + b
}

pub struct Flux2Scheduler {
    /// Длина N+1, последний 0.
    sigmas: Vec<f32>,
}

impl Flux2Scheduler {
    /// Как `retrieve_timesteps(sigmas=linspace(1, 1/N, N), mu=…)`: sigmas
    /// приводятся к f32 (как `np.array(sigmas).astype(np.float32)`), затем
    /// `exp(mu) / (exp(mu) + (1/σ − 1))`.
    pub fn new(num_steps: usize, image_seq_len: usize) -> Self {
        let n = num_steps.max(1);
        let em = empirical_mu(image_seq_len, n).exp();
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let lin = if n == 1 {
                    1.0
                } else {
                    1.0 + i as f64 * ((1.0 / n as f64 - 1.0) / (n - 1) as f64)
                } as f32 as f64;
                (em / (em + (1.0 / lin - 1.0))) as f32
            })
            .collect();
        sigmas.push(0.0);
        Self { sigmas }
    }

    pub fn num_steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    /// img2img: с какого шага начинать при силе `denoise` ∈ (0, 1] (как
    /// `get_timesteps` у diffusers-пайплайнов img2img).
    pub fn start_index(&self, denoise: f32) -> usize {
        let n = self.num_steps() as f64;
        let init = (n * denoise.clamp(0.0, 1.0) as f64).min(n);
        ((n - init).max(0.0) as usize).min(self.num_steps())
    }

    /// `σ·noise + (1 − σ)·x0` при sigma шага `i` (в F32).
    pub fn scale_noise(&self, x0: &Tensor, noise: &Tensor, i: usize) -> Result<Tensor> {
        let s = self.sigmas[i];
        noise
            .to_dtype(DType::F32)?
            .mul_scalar(s)?
            .add(&x0.to_dtype(DType::F32)?.mul_scalar(1.0 - s)?)
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
    fn mu_matches_pipeline() {
        // 512² → 1024 токена латента; значения из diffusers.
        let mu = empirical_mu(1024, 4);
        assert!((mu - 2.030_689_7).abs() < 1e-6, "{mu}");
        assert!((empirical_mu(5000, 50) - (0.00016927 * 5000.0 + 0.45666666)).abs() < 1e-9);
    }

    #[test]
    fn sigmas_match_reference_klein_512() {
        // diffusers: Flux2KleinPipeline, 512×512, 4 шага → scheduler.sigmas.
        let s = Flux2Scheduler::new(4, 1024);
        let expect = [1.0f32, 0.958_085_36, 0.883_981_9, 0.717_496_6, 0.0];
        for (i, e) in expect.iter().enumerate() {
            assert!((s.sigma(i) - e).abs() < 1e-6, "{i}: {} vs {e}", s.sigma(i));
        }
        assert_eq!(s.num_steps(), 4);
    }
}

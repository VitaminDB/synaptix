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
        Self::with_strength(num_steps, image_seq_len, 1.0)
    }

    /// img2img: все `num_steps` шагов, начиная с времени `t₀ = strength` (до
    /// сдвига): `linspace(t₀, t₀/N, N)` → сдвиг → sigmas. При `strength = 1`
    /// совпадает с [`Self::new`].
    ///
    /// Не как `get_timesteps` у img2img-пайплайнов diffusers (отрезать
    /// начальные шаги полного расписания): у klein всего 4 шага, и сила
    /// ступенчатая — 0.5 и 0.7 дают одно и то же. И не «σ = strength»: сдвиг
    /// FLUX.2 сильный (μ ≈ 2 на 4 шагах), композиция решается при σ > 0.9, и
    /// σ = 0.8 уже почти копия исходника. По времени до сдвига шкала ровная:
    /// 0.3 — лёгкая правка, 0.6 — другой стиль при той же композиции, 1 —
    /// картинка заново.
    pub fn with_strength(num_steps: usize, image_seq_len: usize, strength: f32) -> Self {
        let n = num_steps.max(1);
        let em = empirical_mu(image_seq_len, n).exp();
        let t0 = (strength as f64).clamp(1e-4, 1.0) as f32 as f64;
        let mut sigmas: Vec<f32> = (0..n)
            .map(|i| {
                let lin = if n == 1 {
                    t0
                } else {
                    t0 + i as f64 * ((t0 / n as f64 - t0) / (n - 1) as f64)
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

    #[test]
    fn img2img_schedule_starts_at_strength_time() {
        let full = Flux2Scheduler::new(4, 4096);
        let same = Flux2Scheduler::with_strength(4, 4096, 1.0);
        for i in 0..=4 {
            assert_eq!(full.sigma(i), same.sigma(i));
        }
        let s = Flux2Scheduler::with_strength(4, 4096, 0.6);
        assert_eq!(s.num_steps(), 4);
        let em = empirical_mu(4096, 4).exp();
        let want = (em / (em + (1.0 / 0.6f32 as f64 - 1.0))) as f32;
        assert!((s.sigma(0) - want).abs() < 1e-6, "{} vs {want}", s.sigma(0));
        assert!(s.sigma(0) < 1.0);
        for i in 0..4 {
            assert!(s.sigma(i + 1) < s.sigma(i));
        }
        assert_eq!(s.sigma(4), 0.0);
    }
}

//! FlowMatchEulerDiscreteScheduler (FLUX.1-dev) — rectified-flow Euler с
//! dynamic-shifting. Без весов, чистая числовая логика. Bit-exact к diffusers:
//! sigmas = time_shift(mu, linspace(1,1/N,N)); timesteps = sigmas·1000 (длина N);
//! sigmas дополняется 0.0 (длина N+1). step: prev = sample + (σ_next−σ)·v (f32).

use synaptix_core::{dtype::DType, error::Result, tensor::Tensor};

pub struct FlowMatchScheduler {
    sigmas: Vec<f64>,    // длина N+1, последний 0.0
    timesteps: Vec<f32>, // длина N (= sigmas[..N]·1000)
}

impl FlowMatchScheduler {
    /// `calculate_shift`: линейная интерполяция mu по длине latent-последовательности.
    pub fn calculate_shift(image_seq_len: usize) -> f64 {
        let (base_seq, max_seq) = (256.0_f64, 4096.0_f64);
        let (base_shift, max_shift) = (0.5_f64, 1.15_f64);
        let m = (max_shift - base_shift) / (max_seq - base_seq);
        let b = base_shift - m * base_seq;
        image_seq_len as f64 * m + b
    }

    /// `num_steps` шагов, mu из `calculate_shift(image_seq_len)`. num_train=1000.
    pub fn new(num_steps: usize, image_seq_len: usize) -> Self {
        let n = num_steps;
        let mu = Self::calculate_shift(image_seq_len);
        let em = mu.exp();
        // sigmas базовый = linspace(1.0, 1/N, N) в f64, затем time_shift(mu).
        let mut sigmas: Vec<f64> = (0..n)
            .map(|i| {
                let lin = if n == 1 {
                    1.0
                } else {
                    1.0 + i as f64 * ((1.0 / n as f64 - 1.0) / (n - 1) as f64)
                };
                em / (em + (1.0 / lin - 1.0)) // time_shift exponential
            })
            .collect();
        let timesteps: Vec<f32> = sigmas.iter().map(|s| (s * 1000.0) as f32).collect();
        sigmas.push(0.0); // терминальная sigma
        Self { sigmas, timesteps }
    }

    pub fn timesteps(&self) -> &[f32] {
        &self.timesteps
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    /// timestep[i]/1000 = sigma[i] — то, что подаётся в трансформер.
    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i] as f32
    }

    /// Euler-шаг flow-matching: `prev = sample + (σ_{i+1} − σ_i)·v` в f32.
    /// Возвращает f32 (латент держится в f32 между шагами — иначе bf16-квантизация
    /// латента НАКАПЛИВАЕТСЯ за N шагов → зерно; velocity bf16 per-step не копится).
    pub fn step(&self, model_output: &Tensor, i: usize, sample: &Tensor) -> Result<Tensor> {
        let dt = (self.sigmas[i + 1] - self.sigmas[i]) as f32;
        sample
            .to_dtype(DType::F32)?
            .add(&model_output.to_dtype(DType::F32)?.mul_scalar(dt)?)
    }
}

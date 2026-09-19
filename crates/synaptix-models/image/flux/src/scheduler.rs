//! FlowMatchEulerDiscreteScheduler (FLUX.1-dev) — rectified-flow Euler с
//! dynamic-shifting. Без весов, чистая числовая логика. Bit-exact к diffusers:
//! sigmas = time_shift(mu, linspace(1,1/N,N)); timesteps = sigmas·1000 (длина N);
//! sigmas дополняется 0.0 (длина N+1). step: prev = sample + (σ_next−σ)·v (f32).

use synaptix_core::{dtype::DType, error::Result, tensor::Tensor};

/// `scheduler/scheduler_config.json`. dev — динамический сдвиг по длине
/// латентной последовательности, schnell — статический `shift` (= 1.0,
/// то есть без сдвига).
#[derive(Debug, Clone, PartialEq)]
pub struct SchedulerConfig {
    pub use_dynamic_shifting: bool,
    pub shift: f64,
    pub base_image_seq_len: usize,
    pub max_image_seq_len: usize,
    pub base_shift: f64,
    pub max_shift: f64,
}

impl Default for SchedulerConfig {
    /// Значения FLUX.1-dev.
    fn default() -> Self {
        Self {
            use_dynamic_shifting: true,
            shift: 3.0,
            base_image_seq_len: 256,
            max_image_seq_len: 4096,
            base_shift: 0.5,
            max_shift: 1.15,
        }
    }
}

impl SchedulerConfig {
    pub fn from_json(bytes: &[u8]) -> std::result::Result<Self, String> {
        let v: serde_json::Value =
            serde_json::from_slice(bytes).map_err(|e| format!("scheduler_config.json: {e}"))?;
        let d = Self::default();
        let f = |k: &str, def: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(def);
        let u = |k: &str, def: usize| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(def);
        Ok(Self {
            use_dynamic_shifting: v
                .get("use_dynamic_shifting")
                .and_then(|x| x.as_bool())
                .unwrap_or(d.use_dynamic_shifting),
            shift: f("shift", d.shift),
            base_image_seq_len: u("base_image_seq_len", d.base_image_seq_len),
            max_image_seq_len: u("max_image_seq_len", d.max_image_seq_len),
            base_shift: f("base_shift", d.base_shift),
            max_shift: f("max_shift", d.max_shift),
        })
    }

    /// `calculate_shift`: линейная интерполяция mu по длине latent-последовательности.
    pub fn mu(&self, image_seq_len: usize) -> f64 {
        let (base_seq, max_seq) = (self.base_image_seq_len as f64, self.max_image_seq_len as f64);
        let m = (self.max_shift - self.base_shift) / (max_seq - base_seq);
        let b = self.base_shift - m * base_seq;
        image_seq_len as f64 * m + b
    }
}

pub struct FlowMatchScheduler {
    sigmas: Vec<f64>,    // длина N+1, последний 0.0
    timesteps: Vec<f32>, // длина N (= sigmas[..N]·1000)
}

impl FlowMatchScheduler {
    /// `calculate_shift` с константами FLUX.1-dev.
    pub fn calculate_shift(image_seq_len: usize) -> f64 {
        SchedulerConfig::default().mu(image_seq_len)
    }

    /// `num_steps` шагов, mu из `calculate_shift(image_seq_len)`. num_train=1000.
    pub fn new(num_steps: usize, image_seq_len: usize) -> Self {
        Self::with_config(num_steps, image_seq_len, &SchedulerConfig::default())
    }

    /// Как diffusers `FluxPipeline`: базовые sigmas = linspace(1, 1/N, N),
    /// затем экспоненциальный time_shift(mu) либо статический сдвиг.
    pub fn with_config(num_steps: usize, image_seq_len: usize, cfg: &SchedulerConfig) -> Self {
        let n = num_steps.max(1);
        let em = cfg.mu(image_seq_len).exp();
        let mut sigmas: Vec<f64> = (0..n)
            .map(|i| {
                let lin = if n == 1 {
                    1.0
                } else {
                    1.0 + i as f64 * ((1.0 / n as f64 - 1.0) / (n - 1) as f64)
                };
                if cfg.use_dynamic_shifting {
                    em / (em + (1.0 / lin - 1.0)) // time_shift exponential
                } else {
                    cfg.shift * lin / (1.0 + (cfg.shift - 1.0) * lin)
                }
            })
            .collect();
        let timesteps: Vec<f32> = sigmas.iter().map(|s| (s * 1000.0) as f32).collect();
        sigmas.push(0.0); // терминальная sigma
        Self { sigmas, timesteps }
    }

    /// img2img: с какого шага начинать при силе `denoise` ∈ (0, 1]. Как
    /// diffusers `get_timesteps`: `init = min(N·strength, N)`,
    /// `t_start = int(max(N − init, 0))` — усекается разность, а не `init`.
    pub fn start_index(&self, denoise: f32) -> usize {
        let n = self.num_steps() as f64;
        let init = (n * denoise.clamp(0.0, 1.0) as f64).min(n);
        ((n - init).max(0.0) as usize).min(self.num_steps())
    }

    /// `scale_noise`: `σ·noise + (1 − σ)·x0` при sigma шага `i` (в f32).
    pub fn scale_noise(&self, x0: &Tensor, noise: &Tensor, i: usize) -> Result<Tensor> {
        let s = self.sigmas[i] as f32;
        noise
            .to_dtype(DType::F32)?
            .mul_scalar(s)?
            .add(&x0.to_dtype(DType::F32)?.mul_scalar(1.0 - s)?)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_legacy_constructor() {
        let a = FlowMatchScheduler::new(28, 4096);
        let b = FlowMatchScheduler::with_config(28, 4096, &SchedulerConfig::default());
        assert_eq!(a.sigmas, b.sigmas);
        // mu на 4096 токенах = max_shift.
        assert!((SchedulerConfig::default().mu(4096) - 1.15).abs() < 1e-12);
    }

    #[test]
    fn schnell_static_shift_one_is_plain_linspace() {
        let cfg = SchedulerConfig { use_dynamic_shifting: false, shift: 1.0, ..Default::default() };
        let s = FlowMatchScheduler::with_config(4, 4096, &cfg);
        let want = [1.0, 0.75, 0.5, 0.25, 0.0];
        for (i, w) in want.iter().enumerate() {
            assert!((s.sigmas[i] - w).abs() < 1e-12, "sigma[{i}] = {} ≠ {w}", s.sigmas[i]);
        }
    }

    #[test]
    fn config_from_json_reads_schnell() {
        let json = br#"{"use_dynamic_shifting": false, "shift": 1.0, "num_train_timesteps": 1000}"#;
        let cfg = SchedulerConfig::from_json(json).unwrap();
        assert!(!cfg.use_dynamic_shifting);
        assert_eq!(cfg.shift, 1.0);
        assert_eq!(cfg.max_image_seq_len, 4096, "отсутствующие поля — значения dev");
    }

    #[test]
    fn img2img_start_index_truncates_like_diffusers() {
        let s = FlowMatchScheduler::new(28, 4096);
        // diffusers: int(28 − 28·0.6) = int(11.2) = 11.
        assert_eq!(s.start_index(0.6), 11);
        assert_eq!(s.start_index(1.0), 0);
        assert_eq!(s.start_index(0.0), 28);
    }
}

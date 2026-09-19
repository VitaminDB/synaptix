//! `EulerDiscreteScheduler` SDXL-base по `scheduler_config.json`: betas
//! `scaled_linear` 0.00085 → 0.012, шаги `leading` со `steps_offset = 1`,
//! epsilon-предсказание. Для img2img — срез расписания по силе, как
//! `get_timesteps` у `StableDiffusionXLImg2ImgPipeline`.

/// Параметры расписания из `scheduler/scheduler_config.json`.
#[derive(Debug, Clone, PartialEq)]
pub struct EulerParams {
    pub num_train_timesteps: usize,
    pub beta_start: f64,
    pub beta_end: f64,
    pub steps_offset: usize,
    /// `leading` (SDXL-base) или `trailing`.
    pub trailing: bool,
}

impl Default for EulerParams {
    fn default() -> Self {
        Self { num_train_timesteps: 1000, beta_start: 0.00085, beta_end: 0.012, steps_offset: 1, trailing: false }
    }
}

impl EulerParams {
    pub fn from_json(bytes: Option<&[u8]>) -> Self {
        let mut p = Self::default();
        let Some(v) = bytes.and_then(|b| serde_json::from_slice::<serde_json::Value>(b).ok()) else { return p };
        let n = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64());
        p.num_train_timesteps = n("num_train_timesteps").unwrap_or(p.num_train_timesteps);
        p.beta_start = f("beta_start").unwrap_or(p.beta_start);
        p.beta_end = f("beta_end").unwrap_or(p.beta_end);
        p.steps_offset = n("steps_offset").unwrap_or(p.steps_offset);
        p.trailing = v.get("timestep_spacing").and_then(|x| x.as_str()) == Some("trailing");
        p
    }
}

pub struct SdxlEuler {
    /// Моменты шагов (целые, в F32) — от последнего к первому.
    timesteps: Vec<f32>,
    /// Длина N+1, последний 0.
    sigmas: Vec<f32>,
    init_noise_sigma: f32,
}

impl SdxlEuler {
    /// Полное расписание на `steps` шагов.
    pub fn new(steps: usize, p: &EulerParams) -> Self {
        let n = steps.max(1);
        let nt = p.num_train_timesteps;
        // scaled_linear: betas = linspace(√start, √end, T)², в F32 как torch.
        let (a, b) = (p.beta_start.sqrt() as f32, p.beta_end.sqrt() as f32);
        let mut cum = 1f32;
        let sig_full: Vec<f32> = (0..nt)
            .map(|i| {
                let s = a + (b - a) * i as f32 / (nt - 1) as f32;
                cum *= 1.0 - s * s;
                ((1.0 - cum) / cum).sqrt()
            })
            .collect();
        let ts: Vec<usize> = if p.trailing {
            let step = nt as f64 / n as f64;
            (0..n).map(|i| ((nt as f64 - i as f64 * step).round() as i64 - 1).max(0) as usize).collect()
        } else {
            let step = nt / n;
            (0..n).rev().map(|i| (i * step + p.steps_offset).min(nt - 1)).collect()
        };
        let mut sigmas: Vec<f32> = ts.iter().map(|&t| sig_full[t]).collect();
        let smax = sigmas.iter().cloned().fold(0f32, f32::max);
        sigmas.push(0.0);
        Self { timesteps: ts.iter().map(|&t| t as f32).collect(), sigmas, init_noise_sigma: (smax * smax + 1.0).sqrt() }
    }

    /// Первый шаг img2img: `strength` 1 — с начала, 0 — ни одного шага.
    pub fn start_for_strength(&self, strength: f32) -> usize {
        let n = self.num_steps();
        let init = ((n as f32 * strength.clamp(0.0, 1.0)) as usize).min(n);
        n - init
    }

    pub fn num_steps(&self) -> usize {
        self.timesteps.len()
    }

    pub fn timestep(&self, i: usize) -> f32 {
        self.timesteps[i]
    }

    pub fn sigma(&self, i: usize) -> f32 {
        self.sigmas[i]
    }

    pub fn init_noise_sigma(&self) -> f32 {
        self.init_noise_sigma
    }

    /// `x / √(σ² + 1)` перед UNet.
    pub fn input_scale(&self, i: usize) -> f32 {
        1.0 / (self.sigmas[i] * self.sigmas[i] + 1.0).sqrt()
    }

    /// Шаг Эйлера по ε: `x + ε·(σ[i+1] − σ[i])`.
    pub fn dt(&self, i: usize) -> f32 {
        self.sigmas[i + 1] - self.sigmas[i]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_with_offset_like_diffusers() {
        // diffusers 0.40 EulerDiscreteScheduler из каталога SDXL-base, 30 шагов:
        // timesteps [958, 925, 892, …, 1], sigmas [11.47685, 9.54359, …,
        // 0.041314, 0], init_noise_sigma 11.520334.
        let s = SdxlEuler::new(30, &EulerParams::default());
        assert_eq!(s.num_steps(), 30);
        assert_eq!((s.timestep(0), s.timestep(1), s.timestep(29)), (958.0, 925.0, 1.0));
        assert!((s.sigma(0) - 11.476_85).abs() < 1e-3, "{}", s.sigma(0));
        assert!((s.sigma(1) - 9.543_586).abs() < 1e-3, "{}", s.sigma(1));
        assert!((s.sigma(29) - 0.041_314).abs() < 1e-5, "{}", s.sigma(29));
        assert!((s.init_noise_sigma() - 11.520_334).abs() < 1e-3);
        assert_eq!(s.sigma(30), 0.0);
        assert_eq!(s.start_for_strength(1.0), 0);
        assert_eq!(s.start_for_strength(0.5), 15);
        assert_eq!(s.start_for_strength(0.0), 30);
    }
}

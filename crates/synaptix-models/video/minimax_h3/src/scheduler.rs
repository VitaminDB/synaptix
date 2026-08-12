use crate::config::{DEFAULT_SIGMA_SHIFT_AUDIO, DEFAULT_SIGMA_SHIFT_VIDEO};

pub fn shift_sigma(base: f64, shift: f64) -> f64 {
    shift * base / (1.0 + (shift - 1.0) * base)
}

pub fn unshift_sigma(sigma: f64, shift: f64) -> f64 {
    sigma / (shift + sigma * (1.0 - shift))
}

pub fn time_shift_sigma(sigma: f64, from_shift: f64, to_shift: f64) -> f64 {
    shift_sigma(unshift_sigma(sigma, from_shift), to_shift)
}

#[derive(Debug, Clone)]
pub struct H3Scheduler {
    pub sigmas: Vec<f64>,
    pub shift_video: f64,
    pub shift_audio: f64,
}

impl Default for H3Scheduler {
    fn default() -> Self {
        Self::new(20, DEFAULT_SIGMA_SHIFT_VIDEO as f64, DEFAULT_SIGMA_SHIFT_AUDIO as f64)
    }
}

impl H3Scheduler {
    pub fn new(steps: usize, shift_video: f64, shift_audio: f64) -> Self {
        let (ov, oa) = crate::runtime::sigma_shift();
        let shift_video = ov.unwrap_or(shift_video);
        let shift_audio = oa.unwrap_or(shift_audio);
        let steps = steps.max(1);
        let sigmas = (0..=steps)
            .map(|i| {
                let base = 1.0 - i as f64 / steps as f64;
                shift_sigma(base, shift_video)
            })
            .collect();
        Self { sigmas, shift_video, shift_audio }
    }

    pub fn from_sigmas(sigmas: Vec<f64>, shift_video: f64, shift_audio: f64) -> Self {
        Self { sigmas, shift_video, shift_audio }
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len().saturating_sub(1)
    }

    pub fn video_sigma(&self, step: usize) -> f64 {
        self.sigmas[step]
    }

    pub fn audio_sigma(&self, step: usize) -> f64 {
        time_shift_sigma(self.sigmas[step], self.shift_video, self.shift_audio)
    }

    pub fn audio_carry(&self, step: usize) -> f64 {
        let sv = self.sigmas[step].max(1e-6);
        self.audio_sigma(step) / sv
    }

    pub fn video_t(&self, step: usize) -> f64 {
        1.0 - self.video_sigma(step)
    }

    pub fn audio_t(&self, step: usize) -> f64 {
        1.0 - self.audio_sigma(step)
    }

    pub fn video_dt(&self, step: usize) -> f64 {
        self.sigmas[step + 1] - self.sigmas[step]
    }

    pub fn audio_dt(&self, step: usize) -> f64 {
        self.audio_sigma(step + 1) - self.audio_sigma(step)
    }
}

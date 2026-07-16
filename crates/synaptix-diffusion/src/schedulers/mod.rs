use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::{Philox4x32, fill_normal_f32};

use crate::error::{DiffusionError, Result};

pub mod ays;
pub mod consistency;
pub mod ddim;
pub mod ddim_inversion;
pub mod ddpm;
pub mod distilled;
pub mod dpm_pp_2m;
pub mod dpm_pp_3m;
pub mod dpm_pp_sde;
pub mod edm;
pub mod euler;
pub mod euler_a;
pub mod flow_match;
pub mod heun;
pub mod lcm;
pub mod pndm;
pub mod rectified_flow;
pub mod tcd;
pub mod unipc;
pub mod vdm;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PredictionType {
    Epsilon,
    Velocity,
    SampleX0,
    FlowMatchVelocity,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BetaSchedule {
    Linear,
    ScaledLinear,
    SquaredCosCapV2,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestepSpacing {
    Leading,
    Trailing,
    Linspace,
}

#[derive(Debug, Clone)]
pub struct BetaConfig {
    pub num_train_timesteps: usize,
    pub beta_start: f32,
    pub beta_end: f32,
    pub schedule: BetaSchedule,
    pub rescale_zero_snr: bool,
}

impl Default for BetaConfig {
    fn default() -> Self {
        Self {
            num_train_timesteps: 1000,
            beta_start: 0.00085,
            beta_end: 0.012,
            schedule: BetaSchedule::ScaledLinear,
            rescale_zero_snr: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SchedulerOutput {
    pub prev_sample: Tensor,
    pub pred_original_sample: Option<Tensor>,
}

pub trait Scheduler: Send {
    fn set_timesteps(&mut self, n_steps: usize) -> Result<()>;

    fn timesteps(&self) -> &[f32];

    fn sigmas(&self) -> &[f32];

    fn init_noise_sigma(&self) -> f32 {
        self.sigmas().first().copied().unwrap_or(1.0)
    }

    fn prediction_type(&self) -> PredictionType {
        PredictionType::Epsilon
    }

    fn scale_model_input(&self, sample: &Tensor, _step_idx: usize) -> Result<Tensor> {
        Ok(sample.clone())
    }

    fn step(
        &mut self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
    ) -> Result<SchedulerOutput>;

    fn step_with_noise(
        &mut self,
        model_output: &Tensor,
        step_idx: usize,
        sample: &Tensor,
        noise: &Tensor,
    ) -> Result<SchedulerOutput> {
        let _ = noise;
        self.step(model_output, step_idx, sample)
    }

    fn add_noise(&self, original: &Tensor, noise: &Tensor, step_idx: usize) -> Result<Tensor>;

    fn n_steps(&self) -> usize {
        self.sigmas().len().saturating_sub(1)
    }

    fn reset_state(&mut self) {}
}

pub fn betas_for(cfg: &BetaConfig) -> Vec<f32> {
    let n = cfg.num_train_timesteps;
    let mut out = vec![0.0_f32; n];
    match cfg.schedule {
        BetaSchedule::Linear => {
            for (i, b) in out.iter_mut().enumerate() {
                let t = i as f32 / (n.max(2) - 1) as f32;
                *b = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * t;
            }
        }
        BetaSchedule::ScaledLinear => {
            let s = cfg.beta_start.sqrt();
            let e = cfg.beta_end.sqrt();
            for (i, b) in out.iter_mut().enumerate() {
                let t = i as f32 / (n.max(2) - 1) as f32;
                let v = s + (e - s) * t;
                *b = v * v;
            }
        }
        BetaSchedule::SquaredCosCapV2 => {
            let s = 0.008_f32;
            let mut alpha_bar = vec![0.0_f32; n + 1];
            for i in 0..=n {
                let t = i as f32 / n as f32;
                let f = ((t + s) / (1.0 + s) * std::f32::consts::FRAC_PI_2).cos();
                alpha_bar[i] = f * f;
            }
            let base = alpha_bar[0];
            for v in alpha_bar.iter_mut() {
                *v /= base;
            }
            for i in 0..n {
                out[i] = (1.0 - alpha_bar[i + 1] / alpha_bar[i]).clamp(0.0, 0.999);
            }
        }
        BetaSchedule::Sigmoid => {
            for (i, b) in out.iter_mut().enumerate() {
                let t = i as f32 / (n.max(2) - 1) as f32;
                let x = -6.0 + 12.0 * t;
                let s = 1.0 / (1.0 + (-x).exp());
                *b = cfg.beta_start + (cfg.beta_end - cfg.beta_start) * s;
            }
        }
    }
    if cfg.rescale_zero_snr {
        zero_snr_rescale(&mut out);
    }
    out
}

pub fn alphas_cumprod(betas: &[f32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(betas.len());
    let mut acc = 1.0_f32;
    for &b in betas {
        acc *= 1.0 - b;
        out.push(acc);
    }
    out
}

pub fn alphas_to_sigmas(alphas_cum: &[f32]) -> Vec<f32> {
    alphas_cum
        .iter()
        .map(|&a| {
            let a = a.clamp(1e-12, 1.0 - 1e-12);
            ((1.0 - a) / a).sqrt()
        })
        .collect()
}

pub fn sigma_to_alpha(sigma: f32) -> f32 {
    1.0 / (sigma * sigma + 1.0).sqrt()
}

pub fn sigma_to_sigma_data(sigma: f32) -> f32 {
    sigma / (sigma * sigma + 1.0).sqrt()
}

fn zero_snr_rescale(betas: &mut [f32]) {
    let alphas: Vec<f32> = betas.iter().map(|&b| 1.0 - b).collect();
    let mut acum: Vec<f32> = Vec::with_capacity(alphas.len());
    let mut acc = 1.0;
    for &a in &alphas {
        acc *= a;
        acum.push(acc);
    }
    let alphas_bar_sqrt: Vec<f32> = acum.iter().map(|v| v.max(0.0).sqrt()).collect();
    if alphas_bar_sqrt.is_empty() {
        return;
    }
    let last = *alphas_bar_sqrt.last().unwrap();
    let first = alphas_bar_sqrt[0];
    if (first - last).abs() < 1e-12 {
        return;
    }
    let mut scaled: Vec<f32> = alphas_bar_sqrt
        .iter()
        .map(|v| (v - last) * first / (first - last))
        .collect();
    scaled[0] = first;
    let scaled_sq: Vec<f32> = scaled.iter().map(|v| v * v).collect();
    let mut prev = 1.0_f32;
    for (i, b) in betas.iter_mut().enumerate() {
        let cur = scaled_sq[i];
        let alpha = (cur / prev).clamp(1e-12, 1.0);
        *b = (1.0 - alpha).clamp(0.0, 0.999);
        prev = cur;
    }
}

pub fn karras_sigmas(sigma_min: f32, sigma_max: f32, n_steps: usize, rho: f32) -> Vec<f32> {
    if n_steps == 0 {
        return vec![0.0];
    }
    let inv_rho = 1.0 / rho;
    let smax = sigma_max.powf(inv_rho);
    let smin = sigma_min.powf(inv_rho);
    let mut out = Vec::with_capacity(n_steps + 1);
    for i in 0..n_steps {
        let r = i as f32 / (n_steps - 1).max(1) as f32;
        let v = (smax + r * (smin - smax)).powf(rho);
        out.push(v);
    }
    out.push(0.0);
    out
}

pub fn exponential_sigmas(sigma_min: f32, sigma_max: f32, n_steps: usize) -> Vec<f32> {
    if n_steps == 0 {
        return vec![0.0];
    }
    let lmax = sigma_max.ln();
    let lmin = sigma_min.ln();
    let mut out = Vec::with_capacity(n_steps + 1);
    for i in 0..n_steps {
        let r = i as f32 / (n_steps - 1).max(1) as f32;
        out.push((lmax + r * (lmin - lmax)).exp());
    }
    out.push(0.0);
    out
}

pub fn timesteps_from_spacing(
    n_train: usize,
    n_steps: usize,
    spacing: TimestepSpacing,
    offset: usize,
) -> Vec<usize> {
    let n_steps = n_steps.max(1);
    match spacing {
        TimestepSpacing::Leading => {
            let step = n_train / n_steps;
            (0..n_steps)
                .map(|i| (i * step) + offset)
                .rev()
                .collect()
        }
        TimestepSpacing::Trailing => {
            let step = n_train as f32 / n_steps as f32;
            (0..n_steps)
                .map(|i| {
                    let t = (n_train as f32 - (i as f32) * step).round() as i64 - 1;
                    t.max(0) as usize
                })
                .collect()
        }
        TimestepSpacing::Linspace => {
            let last = (n_train - 1) as f32;
            (0..n_steps)
                .map(|i| {
                    let t = last * (1.0 - i as f32 / (n_steps - 1).max(1) as f32);
                    t.round() as usize
                })
                .collect()
        }
    }
}

pub fn sigmas_from_alpha_bar(alphas_bar: &[f32], indices: &[usize]) -> Vec<f32> {
    let all = alphas_to_sigmas(alphas_bar);
    let mut out: Vec<f32> = indices.iter().map(|&i| all[i.min(all.len() - 1)]).collect();
    out.push(0.0);
    out
}

pub fn randn_seeded(shape: &[usize], device: Device, rng: &mut Philox4x32) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut buf = vec![0.0_f32; numel];
    fill_normal_f32(rng, &mut buf);
    Tensor::from_vec(buf, shape.to_vec(), device).map_err(DiffusionError::from)
}

pub fn randn_like(t: &Tensor, rng: &mut Philox4x32) -> Result<Tensor> {
    let buf = randn_seeded(t.dims(), t.device(), rng)?;
    if buf.dtype() == t.dtype() {
        Ok(buf)
    } else {
        cast_tensor(&buf, t.dtype())
    }
}

pub fn cast_tensor(t: &Tensor, dtype: DType) -> Result<Tensor> {
    if t.dtype() == dtype {
        return Ok(t.clone());
    }
    t.to_dtype(dtype).map_err(DiffusionError::from)
}

pub fn convert_to_x0(
    sample: &Tensor,
    model_output: &Tensor,
    sigma: f32,
    prediction_type: PredictionType,
) -> Result<Tensor> {
    match prediction_type {
        PredictionType::SampleX0 => Ok(model_output.clone()),
        PredictionType::Epsilon => {
            let scaled = model_output.affine(sigma, 0.0)?;
            sample.sub(&scaled).map_err(DiffusionError::from)
        }
        PredictionType::Velocity => {
            let alpha = sigma_to_alpha(sigma);
            let sigma_data = sigma_to_sigma_data(sigma);
            let alpha_x = sample.affine(alpha * alpha, 0.0)?;
            let sigma_v = model_output.affine(alpha * sigma_data, 0.0)?;
            alpha_x.sub(&sigma_v).map_err(DiffusionError::from)
        }
        PredictionType::FlowMatchVelocity => {
            let t = sigma;
            let one_minus_t = 1.0 - t;
            let scaled = model_output.affine(t, 0.0)?;
            let denoised = sample.sub(&scaled)?;
            denoised
                .affine(1.0 / one_minus_t.max(1e-12), 0.0)
                .map_err(DiffusionError::from)
        }
    }
}

pub fn convert_to_eps(
    sample: &Tensor,
    model_output: &Tensor,
    sigma: f32,
    prediction_type: PredictionType,
) -> Result<Tensor> {
    match prediction_type {
        PredictionType::Epsilon => Ok(model_output.clone()),
        PredictionType::SampleX0 => {
            let diff = sample.sub(model_output)?;
            diff.affine(1.0 / sigma.max(1e-12), 0.0).map_err(DiffusionError::from)
        }
        PredictionType::Velocity => {
            let alpha = sigma_to_alpha(sigma);
            let sigma_data = sigma_to_sigma_data(sigma);
            let a_v = model_output.affine(alpha, 0.0)?;
            let s_x = sample.affine(sigma_data, 0.0)?;
            a_v.add(&s_x).map_err(DiffusionError::from)
        }
        PredictionType::FlowMatchVelocity => {
            let t = sigma;
            let one_minus_t = 1.0 - t;
            let denoised_x = convert_to_x0(sample, model_output, sigma, PredictionType::FlowMatchVelocity)?;
            let scaled_x0 = denoised_x.affine(one_minus_t, 0.0)?;
            let diff = sample.sub(&scaled_x0)?;
            diff.affine(1.0 / t.max(1e-12), 0.0).map_err(DiffusionError::from)
        }
    }
}

pub fn add_noise_vp(
    original: &Tensor,
    noise: &Tensor,
    sigma: f32,
) -> Result<Tensor> {
    let alpha = sigma_to_alpha(sigma);
    let sigma_data = sigma_to_sigma_data(sigma);
    let scaled_x = original.affine(alpha, 0.0)?;
    let scaled_n = noise.affine(sigma_data, 0.0)?;
    scaled_x.add(&scaled_n).map_err(DiffusionError::from)
}

pub fn add_noise_ve(
    original: &Tensor,
    noise: &Tensor,
    sigma: f32,
) -> Result<Tensor> {
    let scaled = noise.affine(sigma, 0.0)?;
    original.add(&scaled).map_err(DiffusionError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn betas_scaled_linear_matches_sd1_default() {
        let cfg = BetaConfig::default();
        let betas = betas_for(&cfg);
        assert_eq!(betas.len(), 1000);
        assert!((betas[0] - cfg.beta_start).abs() < 1e-6);
        assert!((betas[999] - cfg.beta_end).abs() < 1e-6);
        for w in betas.windows(2) {
            assert!(w[1] >= w[0] - 1e-6);
        }
    }

    #[test]
    fn alphas_cumprod_decreases() {
        let cfg = BetaConfig::default();
        let betas = betas_for(&cfg);
        let acum = alphas_cumprod(&betas);
        assert!(acum[0] > acum[999]);
        assert!(acum[999] > 0.0);
    }

    #[test]
    fn karras_sigmas_monotone_and_zero_terminal() {
        let s = karras_sigmas(0.029, 80.0, 30, 7.0);
        assert_eq!(s.len(), 31);
        assert_eq!(*s.last().unwrap(), 0.0);
        for w in s[..30].windows(2) {
            assert!(w[0] > w[1]);
        }
    }

    #[test]
    fn sigma_alpha_sigma_data_identity() {
        for &s in &[0.1_f32, 1.0, 10.0, 80.0] {
            let a = sigma_to_alpha(s);
            let sd = sigma_to_sigma_data(s);
            assert!((a * a + sd * sd - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn zero_snr_pushes_last_alpha_to_zero() {
        let cfg = BetaConfig {
            rescale_zero_snr: true,
            ..BetaConfig::default()
        };
        let betas = betas_for(&cfg);
        let acum = alphas_cumprod(&betas);
        assert!(acum.last().unwrap().abs() < 1e-4);
    }
}

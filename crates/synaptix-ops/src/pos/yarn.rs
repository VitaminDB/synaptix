use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};

use crate::pos::rope_cache::RopeCache;

#[derive(Debug, Clone, Copy)]
pub struct YarnConfig {
    pub scale: f32,
    pub original_max_seq: usize,
    pub extrapolation_factor: f32,
    pub attn_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl Default for YarnConfig {
    fn default() -> Self {
        Self {
            scale: 1.0,
            original_max_seq: 2048,
            extrapolation_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

pub fn yarn_scaled_rope_cache(
    head_dim: usize,
    max_seq: usize,
    theta_base: f32,
    config: YarnConfig,
    device: Device,
) -> Result<RopeCache> {
    if head_dim % 2 != 0 || head_dim == 0 {
        return Err(SynaptixError::Unsupported("yarn: head_dim must be > 0 and even"));
    }
    if config.scale <= 0.0 || config.original_max_seq == 0 {
        return Err(SynaptixError::Unsupported("yarn: invalid config"));
    }
    let half = head_dim / 2;
    let log_base = theta_base.ln();
    let mut freqs = Vec::with_capacity(half);
    let two_pi = 2.0 * std::f32::consts::PI;
    let low_idx = (head_dim as f32 * (config.original_max_seq as f32 / (two_pi * config.beta_fast)).ln() / log_base) / 2.0;
    let high_idx = (head_dim as f32 * (config.original_max_seq as f32 / (two_pi * config.beta_slow)).ln() / log_base) / 2.0;
    for i in 0..half {
        let exponent = -(2.0 * i as f32) / (head_dim as f32);
        let base_freq = theta_base.powf(exponent);
        let scaled_freq = base_freq / config.scale;
        let mut ramp = (i as f32 - low_idx) / (high_idx - low_idx);
        if ramp < 0.0 { ramp = 0.0; }
        if ramp > 1.0 { ramp = 1.0; }
        let mix = (1.0 - ramp) * scaled_freq + ramp * base_freq;
        freqs.push(mix);
    }
    RopeCache::with_scaled_freqs(head_dim, max_seq, theta_base, &freqs, device)
}

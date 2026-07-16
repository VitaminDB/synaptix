use synaptix_core::device::Device;
use synaptix_core::error::{Result, SynaptixError};

use crate::pos::rope_cache::RopeCache;

#[derive(Debug, Clone)]
pub struct LongRopeConfig {
    pub long_factors: Vec<f32>,
    pub short_factors: Vec<f32>,
    pub original_max_seq: usize,
}

pub fn longrope_cache(
    head_dim: usize,
    max_seq: usize,
    theta_base: f32,
    config: &LongRopeConfig,
    device: Device,
) -> Result<RopeCache> {
    if head_dim % 2 != 0 || head_dim == 0 {
        return Err(SynaptixError::Unsupported("longrope: head_dim must be > 0 and even"));
    }
    let half = head_dim / 2;
    if config.long_factors.len() != half || config.short_factors.len() != half {
        return Err(SynaptixError::Other(format!(
            "longrope: factor length mismatch (expected {half})"
        )));
    }
    let use_long = max_seq > config.original_max_seq;
    let factors = if use_long { &config.long_factors } else { &config.short_factors };
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = -(2.0 * i as f32) / (head_dim as f32);
        let base_freq = theta_base.powf(exponent);
        freqs.push(base_freq / factors[i]);
    }
    RopeCache::with_scaled_freqs(head_dim, max_seq, theta_base, &freqs, device)
}

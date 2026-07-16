use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct RopeCache {
    cos: Tensor,
    sin: Tensor,
    theta_base: f32,
    head_dim: usize,
    max_seq: usize,
}

impl RopeCache {
    pub fn new(head_dim: usize, max_seq: usize, theta_base: f32, device: Device) -> Result<Self> {
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(SynaptixError::Unsupported("RopeCache: head_dim must be > 0 and even"));
        }
        let half = head_dim / 2;
        let mut freqs = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = -(2.0 * i as f32) / (head_dim as f32);
            freqs.push(theta_base.powf(exponent));
        }
        let mut cos = vec![0.0_f32; max_seq * half];
        let mut sin = vec![0.0_f32; max_seq * half];
        for t in 0..max_seq {
            for i in 0..half {
                let angle = (t as f32) * freqs[i];
                cos[t * half + i] = angle.cos();
                sin[t * half + i] = angle.sin();
            }
        }
        let cos_t = Tensor::from_vec(cos, (max_seq, half), device)?;
        let sin_t = Tensor::from_vec(sin, (max_seq, half), device)?;
        Ok(Self { cos: cos_t, sin: sin_t, theta_base, head_dim, max_seq })
    }

    pub fn with_scaled_freqs(
        head_dim: usize,
        max_seq: usize,
        theta_base: f32,
        scaled_freqs: &[f32],
        device: Device,
    ) -> Result<Self> {
        if head_dim == 0 || head_dim % 2 != 0 {
            return Err(SynaptixError::Unsupported("RopeCache: head_dim must be > 0 and even"));
        }
        let half = head_dim / 2;
        if scaled_freqs.len() != half {
            return Err(SynaptixError::Other(format!(
                "scaled_freqs len {} mismatch head_dim/2 {}",
                scaled_freqs.len(),
                half
            )));
        }
        let mut cos = vec![0.0_f32; max_seq * half];
        let mut sin = vec![0.0_f32; max_seq * half];
        for t in 0..max_seq {
            for i in 0..half {
                let angle = (t as f32) * scaled_freqs[i];
                cos[t * half + i] = angle.cos();
                sin[t * half + i] = angle.sin();
            }
        }
        let cos_t = Tensor::from_vec(cos, (max_seq, half), device)?;
        let sin_t = Tensor::from_vec(sin, (max_seq, half), device)?;
        Ok(Self { cos: cos_t, sin: sin_t, theta_base, head_dim, max_seq })
    }

    pub fn cos(&self) -> &Tensor { &self.cos }
    pub fn sin(&self) -> &Tensor { &self.sin }
    pub fn head_dim(&self) -> usize { self.head_dim }
    pub fn max_seq(&self) -> usize { self.max_seq }
    pub fn theta_base(&self) -> f32 { self.theta_base }

    pub fn select_positions(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let cos_sel = self.cos.index_select(0, positions)?;
        let sin_sel = self.sin.index_select(0, positions)?;
        Ok((cos_sel, sin_sel))
    }

    pub fn select_range(&self, start: usize, len: usize) -> Result<(Tensor, Tensor)> {
        let cos_sel = self.cos.narrow(0, start, len)?.contiguous()?;
        let sin_sel = self.sin.narrow(0, start, len)?.contiguous()?;
        Ok((cos_sel, sin_sel))
    }
}

pub fn build_default_cache(
    head_dim: usize,
    max_seq: usize,
    theta_base: f32,
    device: Device,
) -> Result<RopeCache> {
    let _ = DType::F32;
    RopeCache::new(head_dim, max_seq, theta_base, device)
}

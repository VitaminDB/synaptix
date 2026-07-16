use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// Diffusion-style time embedding: sinusoidal-эмбеддинг шага шумa → fc1 → SiLU → fc2.
///
/// Schema HF `diffusers.get_timestep_embedding` (flip_sin_to_cos=true,
/// downscale_freq_shift=1, scale=1):
///
/// ```text
/// half = in_dim / 2
/// exponent = −ln(10000) · arange(half) / (half − 1)
/// freqs = exp(exponent)                                        # [half]
/// args = t.unsqueeze(-1) · freqs.unsqueeze(0)                   # [B, half]
/// emb = cat([cos(args), sin(args)], dim=-1)                    # [B, in_dim]
/// ```
///
/// `in_dim` должен быть чётным.
pub struct TimeEmbedding {
    pub fc1: Linear,
    pub fc2: Linear,
    pub in_dim: usize,
    pub out_dim: usize,
}

impl TimeEmbedding {
    pub fn new(
        in_dim: usize,
        hidden_dim: usize,
        out_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        if in_dim % 2 != 0 {
            return Err(SynaptixError::Unsupported("TimeEmbedding: in_dim must be even"));
        }
        Ok(Self {
            fc1: Linear::from_init(
                in_dim, hidden_dim, true,
                InitMethod::KaimingUniform { fan_in: in_dim, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            fc2: Linear::from_init(
                hidden_dim, out_dim, true,
                InitMethod::KaimingUniform { fan_in: hidden_dim, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            in_dim,
            out_dim,
        })
    }

    pub fn from_weights(
        fc1_w: Tensor, fc1_b: Option<Tensor>,
        fc2_w: Tensor, fc2_b: Option<Tensor>,
    ) -> Result<Self> {
        let fc1 = Linear::new(fc1_w, fc1_b)?;
        let fc2 = Linear::new(fc2_w, fc2_b)?;
        Ok(Self {
            in_dim: fc1.in_features(),
            out_dim: fc2.out_features(),
            fc1,
            fc2,
        })
    }

    /// `timesteps: [B]` (F32 на CPU, любая dtype на target device). Возвращает `[B, out_dim]`.
    pub fn forward(&self, timesteps: &Tensor) -> Result<Tensor> {
        let emb = sinusoidal_timestep_embedding(timesteps, self.in_dim)?;
        let emb_dtype = emb.to_dtype(self.fc1.weight().dtype())?;
        let h = self.fc1.forward(&emb_dtype)?;
        let h = h.silu()?;
        self.fc2.forward(&h)
    }
}

/// HF diffusers timestep embedding (flip_sin_to_cos=true, downscale_freq_shift=1).
pub fn sinusoidal_timestep_embedding(timesteps: &Tensor, in_dim: usize) -> Result<Tensor> {
    if in_dim % 2 != 0 {
        return Err(SynaptixError::Unsupported("sinusoidal_timestep_embedding: in_dim must be even"));
    }
    if timesteps.rank() != 1 {
        return Err(SynaptixError::Unsupported("sinusoidal_timestep_embedding: timesteps must be 1D [B]"));
    }
    let device = timesteps.device();
    let half = in_dim / 2;
    let denom = (half as f32) - 1.0;
    let log10000 = 10000.0_f32.ln();
    let mut freqs = Vec::with_capacity(half);
    for i in 0..half {
        let exponent = -log10000 * (i as f32) / denom.max(1.0);
        freqs.push(exponent.exp());
    }
    let freqs_t = Tensor::from_vec(freqs, (1, half), device)?.to_dtype(DType::F32)?;
    let t_f32 = timesteps.to_dtype(DType::F32)?;
    let t_col = t_f32.reshape(vec![timesteps.dims()[0], 1])?;
    let args = t_col.broadcast_mul(&freqs_t)?;
    let cos = args.cos()?.contiguous()?;
    let sin = args.sin()?.contiguous()?;
    Tensor::cat(&[&cos, &sin], 1)
}

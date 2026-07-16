use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::multimodal::cross_modal::CrossModalAttention;
use crate::parameter::Parameter;

/// Perceiver Resampler (Jaegle et al. / Flamingo style) — обучаемые
/// `num_latents` latent-токенов cross-attend'ятся к произвольно-длинному
/// context (variable-length token stream). На выходе фиксированное число
/// токенов — независимо от длины input'а.
///
/// Минимально: один cross-attention слой. Полный Flamingo Perceiver — N
/// слоёв с self-attention между latents — Phase O.
pub struct PerceiverResampler {
    pub latents: Parameter,
    pub cross_attn: CrossModalAttention,
    pub num_latents: usize,
    pub hidden_size: usize,
    pub context_dim: usize,
}

impl PerceiverResampler {
    pub fn new(
        num_latents: usize, hidden_size: usize, context_dim: usize,
        num_heads: usize, device: Device, dtype: DType,
    ) -> Result<Self> {
        let lat = crate::init::init_tensor(
            &[num_latents, hidden_size],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype, 0, device,
        )?;
        Ok(Self {
            latents: Parameter::new(lat),
            cross_attn: CrossModalAttention::new(
                hidden_size, context_dim, num_heads, device, dtype,
            )?,
            num_latents,
            hidden_size,
            context_dim,
        })
    }

    pub fn from_weights(latents: Tensor, cross_attn: CrossModalAttention) -> Result<Self> {
        if latents.rank() != 2 {
            return Err(SynaptixError::Unsupported(
                "PerceiverResampler: latents must be [num_latents, hidden_size]",
            ));
        }
        let num_latents = latents.dims()[0];
        let hidden_size = latents.dims()[1];
        let context_dim = cross_attn.context_dim;
        Ok(Self {
            latents: Parameter::new(latents),
            cross_attn,
            num_latents,
            hidden_size,
            context_dim,
        })
    }

    /// `context: [B, Sk, context_dim]` → `[B, num_latents, hidden_size]`.
    pub fn forward(&self, context: &Tensor) -> Result<Tensor> {
        if context.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "PerceiverResampler: context must be [B, Sk, context_dim]",
            ));
        }
        let batch = context.dims()[0];
        let latents_b = self.latents.tensor()
            .reshape(vec![1, self.num_latents, self.hidden_size])?
            .broadcast_mul(&Tensor::ones(
                vec![batch, self.num_latents, self.hidden_size],
                context.dtype(), context.device(),
            )?)?;
        self.cross_attn.forward(&latents_b, context, None)
    }
}

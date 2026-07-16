use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::multimodal::cross_modal::CrossModalAttention;
use crate::parameter::Parameter;

/// BLIP-2 Q-Former (Li et al. 2023) — обучаемые `num_query_tokens`
/// query-токенов, которые cross-attend'ятся к image-features. Возвращает
/// сжатое представление [B, num_query_tokens, hidden_size].
///
/// Минимальная реализация: один cross-attention слой. Полный BLIP-2
/// Q-Former — N×(self-attn + cross-attn + FFN) — откладывается до Phase O.
pub struct QFormer {
    pub query_tokens: Parameter,
    pub cross_attn: CrossModalAttention,
    pub num_query_tokens: usize,
    pub hidden_size: usize,
    pub context_dim: usize,
}

impl QFormer {
    pub fn new(
        num_query_tokens: usize, hidden_size: usize, context_dim: usize,
        num_heads: usize, device: Device, dtype: DType,
    ) -> Result<Self> {
        let q = crate::init::init_tensor(
            &[num_query_tokens, hidden_size],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype, 0, device,
        )?;
        Ok(Self {
            query_tokens: Parameter::new(q),
            cross_attn: CrossModalAttention::new(
                hidden_size, context_dim, num_heads, device, dtype,
            )?,
            num_query_tokens,
            hidden_size,
            context_dim,
        })
    }

    pub fn from_weights(
        query_tokens: Tensor, cross_attn: CrossModalAttention,
    ) -> Result<Self> {
        if query_tokens.rank() != 2 {
            return Err(SynaptixError::Unsupported(
                "QFormer: query_tokens must be [num_query_tokens, hidden_size]",
            ));
        }
        let num_query_tokens = query_tokens.dims()[0];
        let hidden_size = query_tokens.dims()[1];
        let context_dim = cross_attn.context_dim;
        Ok(Self {
            query_tokens: Parameter::new(query_tokens),
            cross_attn,
            num_query_tokens,
            hidden_size,
            context_dim,
        })
    }

    /// `image_features: [B, Sk, context_dim]` → `[B, num_query_tokens, hidden_size]`.
    pub fn forward(&self, image_features: &Tensor) -> Result<Tensor> {
        if image_features.rank() != 3 {
            return Err(SynaptixError::Unsupported(
                "QFormer: image_features must be [B, Sk, context_dim]",
            ));
        }
        let batch = image_features.dims()[0];
        let q = self.query_tokens.tensor()
            .reshape(vec![1, self.num_query_tokens, self.hidden_size])?
            .broadcast_mul(&Tensor::ones(
                vec![batch, self.num_query_tokens, self.hidden_size],
                image_features.dtype(), image_features.device(),
            )?)?;
        self.cross_attn.forward(&q, image_features, None)
    }
}

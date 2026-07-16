use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// SpeakerEncoder — минимальная ECAPA-TDNN / x-vector обёртка.
///
/// Архитектура:
///   `x: [B, T, in_channels] →
///        TDNN1 (in→h, ReLU) → TDNN2 (h→h, ReLU) → TDNN3 (h→h, ReLU) →
///        statistical pooling по T (mean+std concat) →
///        emb_proj (2h → embedding_size)`.
///
/// Реальная ECAPA-TDNN использует dilated conv1d (kernel>1) + SE-блоки +
/// attentive pooling — отложено в Phase O. Здесь TDNN-слои = Linear (kernel=1)
/// + ReLU; pooling = mean+std (statistical, без attention).
pub struct SpeakerEncoder {
    pub tdnn1: Linear,
    pub tdnn2: Linear,
    pub tdnn3: Linear,
    pub emb_proj: Linear,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub embedding_size: usize,
}

impl SpeakerEncoder {
    pub fn new(
        in_channels: usize, hidden_size: usize, embedding_size: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            tdnn2: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            tdnn3: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            emb_proj: Linear::from_init(
                hidden_size * 2, embedding_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size * 2, a: 0.0 },
                InitMethod::Zeros, device, dtype, 3,
            )?,
            in_channels,
            hidden_size,
            embedding_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 {
            return Err(SynaptixError::Unsupported("SpeakerEncoder: x must be [B, T, in_channels]"));
        }
        let h1 = self.tdnn1.forward(x)?.relu()?;
        let h2 = self.tdnn2.forward(&h1)?.relu()?;
        let h3 = self.tdnn3.forward(&h2)?.relu()?;
        let mean = h3.mean_keepdim(1)?;
        let centered = h3.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(1)?;
        let std = var.add_scalar(1e-9)?.sqrt()?;
        let mean_s = mean.squeeze(1)?;
        let std_s = std.squeeze(1)?;
        let pooled = Tensor::cat(&[&mean_s, &std_s], 1)?;
        self.emb_proj.forward(&pooled)
    }
}

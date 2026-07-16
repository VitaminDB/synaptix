use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::unet::attn_block::UNetAttnBlock;
use crate::unet::resnet_block::ResNetBlock;
use crate::unet::time_embedding::TimeEmbedding;

/// Минимальный UNet3D pipeline для video: `[B, T, S, in_channels]` →
/// flatten в `[B, T·S, in_channels]` → conv_in → ResNet(time_emb) → temporal
/// self-attention → conv_out → reshape обратно в `[B, T, S, out_channels]`.
///
/// Temporal-attention выполняется по объединённой `T·S`-оси: для bit-exact
/// тестов этого достаточно, реальное разделение spatial/temporal attention
/// выносится на caller.
pub struct UNet3d {
    pub conv_in: Linear,
    pub time_embedding: TimeEmbedding,
    pub resnet: ResNetBlock,
    pub temporal_attn: UNetAttnBlock,
    pub conv_out: Linear,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub out_channels: usize,
}

impl UNet3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_size: usize,
        num_heads: usize,
        time_in_dim: usize,
        time_hidden_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            conv_in: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            time_embedding: TimeEmbedding::new(time_in_dim, time_hidden_dim, hidden_size, device, dtype)?,
            resnet: ResNetBlock::new(hidden_size, hidden_size, hidden_size, device, dtype)?,
            temporal_attn: UNetAttnBlock::new(hidden_size, num_heads, device, dtype)?,
            conv_out: Linear::from_init(
                hidden_size, out_channels, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            in_channels,
            hidden_size,
            out_channels,
        })
    }

    /// `x: [B, T, S, in_channels]`, `timesteps: [B]`.
    pub fn forward(&self, x: &Tensor, timesteps: &Tensor) -> Result<Tensor> {
        if x.rank() != 4 || x.dims()[3] != self.in_channels {
            return Err(SynaptixError::Unsupported("UNet3d: expects x [B, T, S, in_channels]"));
        }
        let b = x.dims()[0];
        let t = x.dims()[1];
        let s = x.dims()[2];
        let x_flat = x.reshape(vec![b, t * s, self.in_channels])?;
        let h = self.conv_in.forward(&x_flat)?;
        let t_emb = self.time_embedding.forward(timesteps)?;
        let h = self.resnet.forward(&h, &t_emb)?;
        let h = self.temporal_attn.forward(&h)?;
        let h = self.conv_out.forward(&h)?;
        h.reshape(vec![b, t, s, self.out_channels])
    }
}

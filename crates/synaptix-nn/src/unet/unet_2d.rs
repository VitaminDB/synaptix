use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::unet::attn_block::UNetAttnBlock;
use crate::unet::cross_attn_block::UNetCrossAttnBlock;
use crate::unet::resnet_block::ResNetBlock;
use crate::unet::time_embedding::TimeEmbedding;

/// Минимальный UNet2D pipeline (Linear-stub: spatial-измерения уже flatten'нуты
/// в seq):
///
/// `conv_in → ResNet(time_emb) → UNetAttnBlock → UNetCrossAttnBlock(text_ctx) → conv_out`.
///
/// `forward(x: [B, T, in_channels], timesteps: [B], text_ctx: [B, S_ctx, context_dim])`.
pub struct UNet2d {
    pub conv_in: Linear,
    pub time_embedding: TimeEmbedding,
    pub resnet: ResNetBlock,
    pub attn: UNetAttnBlock,
    pub cross_attn: UNetCrossAttnBlock,
    pub conv_out: Linear,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub out_channels: usize,
}

impl UNet2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        hidden_size: usize,
        num_heads: usize,
        context_dim: usize,
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
            attn: UNetAttnBlock::new(hidden_size, num_heads, device, dtype)?,
            cross_attn: UNetCrossAttnBlock::new(hidden_size, context_dim, num_heads, device, dtype)?,
            conv_out: Linear::from_init(
                hidden_size, out_channels, true,
                InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1,
            )?,
            in_channels,
            hidden_size,
            out_channels,
        })
    }

    pub fn forward(&self, x: &Tensor, timesteps: &Tensor, text_ctx: &Tensor) -> Result<Tensor> {
        if x.rank() != 3 || x.dims()[2] != self.in_channels {
            return Err(SynaptixError::Unsupported("UNet2d: expects x [B, T, in_channels]"));
        }
        let h = self.conv_in.forward(x)?;
        let t_emb = self.time_embedding.forward(timesteps)?;
        let h = self.resnet.forward(&h, &t_emb)?;
        let h = self.attn.forward(&h)?;
        let h = self.cross_attn.forward(&h, text_ctx)?;
        self.conv_out.forward(&h)
    }
}

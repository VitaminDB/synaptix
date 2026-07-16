use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::vision::vit::VisionTransformer;

/// SAM (Segment Anything) ViT-H image encoder (Kirillov et al. 2023).
///
/// Compose: `ViT-H` (без CLS) → `neck` (Linear hidden→neck_dim, обычно 256).
/// Полная SAM использует window-attention с relative position bias —
/// откладывается до Phase O (здесь обычный full self-attn ViT). Выход —
/// токены [B, N, neck_dim] для mask decoder.
pub struct SamVit {
    pub vit: VisionTransformer,
    pub neck: Linear,
    pub hidden_size: usize,
    pub neck_dim: usize,
}

impl SamVit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize, patch_size: usize, image_size: usize,
        hidden_size: usize, num_heads: usize, num_layers: usize, ffn_dim: usize,
        neck_dim: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            vit: VisionTransformer::new(
                in_channels, patch_size, image_size,
                hidden_size, num_heads, num_layers, ffn_dim,
                device, dtype,
            )?,
            neck: Linear::from_init(
                hidden_size, neck_dim, false,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: neck_dim },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            hidden_size,
            neck_dim,
        })
    }

    /// `image: [B, C, H, W]` → `tokens: [B, num_patches, neck_dim]`.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let hidden = self.vit.forward(image)?;
        self.neck.forward(&hidden)
    }
}

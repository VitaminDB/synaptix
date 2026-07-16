use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::pooling::mean_pool;
use crate::vision::vit::VisionTransformer;

/// NaViT — Native Aspect Ratio ViT (Dehghani et al. 2023).
///
/// Обрабатывает изображения **с произвольным H×W** (не квадратные) при
/// фиксированном `patch_size`. Архитектура = ViT, но не требует resize до
/// фикс. квадрата. `patch_position_embedding` (factorized по h/w) и
/// `pack-and-mask` для batch'а разнокалиберных изображений — Phase O.
/// Здесь — обычный ViT-forward (он уже принимает любой H/W кратный
/// patch_size).
pub struct NaViT {
    pub vit: VisionTransformer,
    pub hidden_size: usize,
}

impl NaViT {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize, patch_size: usize, image_size: usize,
        hidden_size: usize, num_heads: usize, num_layers: usize, ffn_dim: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            vit: VisionTransformer::new(
                in_channels, patch_size, image_size,
                hidden_size, num_heads, num_layers, ffn_dim,
                device, dtype,
            )?,
            hidden_size,
        })
    }

    /// `image: [B, C, H, W]` (H и W могут быть разными, кратными patch_size)
    /// → `embedding: [B, hidden_size]` (mean-pool).
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        if image.rank() != 4 {
            return Err(SynaptixError::Unsupported("NaViT: image must be [B, C, H, W]"));
        }
        let hidden = self.vit.forward(image)?;
        let seq_dim = hidden.rank() - 2;
        mean_pool(&hidden, seq_dim)
    }

    pub fn forward_features(&self, image: &Tensor) -> Result<Tensor> {
        self.vit.forward(image)
    }
}

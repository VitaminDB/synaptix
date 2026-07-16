use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::pooling::cls_pool;
use crate::vision::vit::VisionTransformer;

/// CLIP ViT image encoder.
///
/// `image → patchify → ViT blocks → final_ln → CLS-pool → visual_projection`.
/// Совместима с HuggingFace `transformers.CLIPVisionModel` по структуре:
/// pre-norm ViT с CLS-токеном (CLS добавляется на patchify-уровне в полной
/// реализации, здесь — `cls_pool` берёт первый токен из последовательности).
/// `visual_projection`: Linear hidden→embed_dim (для contrastive alignment).
pub struct ClipVision {
    pub vit: VisionTransformer,
    pub visual_projection: Linear,
    pub hidden_size: usize,
    pub embed_dim: usize,
}

impl ClipVision {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize, patch_size: usize, image_size: usize,
        hidden_size: usize, num_heads: usize, num_layers: usize, ffn_dim: usize,
        embed_dim: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            vit: VisionTransformer::new(
                in_channels, patch_size, image_size,
                hidden_size, num_heads, num_layers, ffn_dim,
                device, dtype,
            )?,
            visual_projection: Linear::from_init(
                hidden_size, embed_dim, false,
                InitMethod::XavierUniform { fan_in: hidden_size, fan_out: embed_dim },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            hidden_size,
            embed_dim,
        })
    }

    pub fn from_parts(
        vit: VisionTransformer, visual_projection: Linear,
    ) -> Result<Self> {
        let hidden_size = vit.hidden_size;
        let embed_dim = visual_projection.out_features();
        Ok(Self { vit, visual_projection, hidden_size, embed_dim })
    }

    /// `image: [B, C, H, W]` → `embedding: [B, embed_dim]`.
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let hidden = self.vit.forward(image)?;
        let pooled = cls_pool(&hidden)?;
        self.visual_projection.forward(&pooled)
    }

    /// Возвращает все patch-токены без pooling и projection.
    pub fn forward_features(&self, image: &Tensor) -> Result<Tensor> {
        self.vit.forward(image)
    }
}


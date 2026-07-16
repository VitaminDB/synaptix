use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::pooling::mean_pool;
use crate::vision::vit::VisionTransformer;

/// SigLIP vision encoder (Zhai et al. 2023).
///
/// Архитектурно совпадает с CLIP-ViT, отличия: (1) sigmoid-based contrastive
/// loss (живёт в loss-функции, не в encoder); (2) **MAP-head** или mean-pool
/// вместо CLS-токена; (3) выход не L2-нормализуется внутри encoder'а — это
/// делается в contrastive head. Здесь `forward` возвращает hidden state после
/// mean-pool + visual_projection.
pub struct SigLip {
    pub vit: VisionTransformer,
    pub visual_projection: Linear,
    pub hidden_size: usize,
    pub embed_dim: usize,
}

impl SigLip {
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

    pub fn from_parts(vit: VisionTransformer, visual_projection: Linear) -> Result<Self> {
        let hidden_size = vit.hidden_size;
        let embed_dim = visual_projection.out_features();
        Ok(Self { vit, visual_projection, hidden_size, embed_dim })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let hidden = self.vit.forward(image)?;
        let seq_dim = hidden.rank() - 2;
        let pooled = mean_pool(&hidden, seq_dim)?;
        self.visual_projection.forward(&pooled)
    }

    pub fn forward_features(&self, image: &Tensor) -> Result<Tensor> {
        self.vit.forward(image)
    }
}

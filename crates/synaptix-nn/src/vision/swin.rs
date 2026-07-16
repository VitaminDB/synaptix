use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;
use crate::pooling::mean_pool;
use crate::vision::vit::ViTBlock;

/// Swin Transformer (Liu et al. 2021) — иерархический ViT с window-attention
/// и `PatchMerging` между stage'ами.
///
/// **Упрощённая реализация:** stage'ы используют обычные `ViTBlock` (full
/// self-attn), без window partition + shift. Полный window-attention с
/// relative position bias откладывается до Phase O. PatchMerging:
/// `[B, H, W, C] → [B, H/2, W/2, 4C] → Linear(4C → 2C)` корректен.
pub struct Swin {
    pub patch_embed: Linear,
    pub stages: Vec<SwinStage>,
    pub norm_w: Parameter,
    pub norm_b: Parameter,
    pub patch_size: usize,
    pub in_channels: usize,
    pub hidden_size: usize,
}

/// Один stage: блоки + опциональный PatchMerging в конце.
pub struct SwinStage {
    pub blocks: Vec<ViTBlock>,
    pub merge: Option<Linear>,
    pub merge_norm_w: Option<Parameter>,
    pub merge_norm_b: Option<Parameter>,
}

impl Swin {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize, patch_size: usize,
        hidden_size: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let patch_dim = patch_size * patch_size * in_channels;
        let norm_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let norm_b = Tensor::zeros(vec![hidden_size], dtype, device)?;
        Ok(Self {
            patch_embed: Linear::from_init(
                patch_dim, hidden_size, true,
                InitMethod::XavierUniform { fan_in: patch_dim, fan_out: hidden_size },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            stages: Vec::new(),
            norm_w: Parameter::new(norm_w),
            norm_b: Parameter::new(norm_b),
            patch_size,
            in_channels,
            hidden_size,
        })
    }

    fn patchify(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 4 {
            return Err(SynaptixError::Unsupported("Swin: image must be [B, C, H, W]"));
        }
        let (b, c, h, w) = (x.dims()[0], x.dims()[1], x.dims()[2], x.dims()[3]);
        let p = self.patch_size;
        if h % p != 0 || w % p != 0 {
            return Err(SynaptixError::Unsupported("Swin: H/W должны делиться на patch_size"));
        }
        let nh = h / p;
        let nw = w / p;
        let reshaped = x.reshape(vec![b, c, nh, p, nw, p])?;
        let permuted = reshaped.permute(vec![0, 2, 4, 1, 3, 5])?.contiguous()?;
        permuted.reshape(vec![b, nh * nw, c * p * p])
    }

    /// `image: [B, C, H, W]` → `embedding: [B, hidden_size_final]` (mean-pool).
    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        let tokens = self.patchify(image)?;
        let mut h = self.patch_embed.forward(&tokens)?;
        for stage in &self.stages {
            for block in &stage.blocks {
                h = block.forward(&h)?;
            }
            if let Some(m) = stage.merge.as_ref() {
                let (b_n, s_n, c_n) = (h.dims()[0], h.dims()[1], h.dims()[2]);
                if s_n % 4 != 0 {
                    return Err(SynaptixError::Unsupported(
                        "Swin PatchMerging: длина токенов не кратна 4",
                    ));
                }
                let merged = h.reshape(vec![b_n, s_n / 4, c_n * 4])?;
                let normed = if let (Some(w), Some(b)) =
                    (stage.merge_norm_w.as_ref(), stage.merge_norm_b.as_ref())
                {
                    layer_norm(&merged, Some(&w.tensor()), Some(&b.tensor()), 1e-5)?
                } else {
                    merged
                };
                h = m.forward(&normed)?;
            }
        }
        let final_normed = layer_norm(
            &h,
            Some(&self.norm_w.tensor()),
            Some(&self.norm_b.tensor()),
            1e-5,
        )?;
        let seq_dim = final_normed.rank() - 2;
        mean_pool(&final_normed, seq_dim)
    }
}

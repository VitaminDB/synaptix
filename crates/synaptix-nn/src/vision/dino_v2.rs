use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::parameter::Parameter;
use crate::pooling::cls_pool;
use crate::vision::vit::VisionTransformer;

/// DINOv2 (Oquab et al. 2024) — self-supervised ViT с **register tokens**.
///
/// Регистровые токены (Darcet et al. 2024) добавляются между CLS и
/// patch-токенами для стабильности attention-карт. Здесь они хранятся как
/// обучаемые `Parameter` shape `[num_registers, hidden_size]` и prepend'ятся
/// к patch-последовательности после patchify. Pool по CLS (первый токен).
pub struct DinoV2 {
    pub vit: VisionTransformer,
    pub cls_token: Parameter,
    pub register_tokens: Option<Parameter>,
    pub num_registers: usize,
    pub hidden_size: usize,
}

impl DinoV2 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize, patch_size: usize, image_size: usize,
        hidden_size: usize, num_heads: usize, num_layers: usize, ffn_dim: usize,
        num_registers: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let cls = crate::init::init_tensor(
            &[1, hidden_size],
            InitMethod::Normal { mean: 0.0, std: 0.02 },
            dtype, 0, device,
        )?;
        let registers = if num_registers > 0 {
            Some(Parameter::new(crate::init::init_tensor(
                &[num_registers, hidden_size],
                InitMethod::Normal { mean: 0.0, std: 0.02 },
                dtype, 1, device,
            )?))
        } else {
            None
        };
        Ok(Self {
            vit: VisionTransformer::new(
                in_channels, patch_size, image_size,
                hidden_size, num_heads, num_layers, ffn_dim,
                device, dtype,
            )?,
            cls_token: Parameter::new(cls),
            register_tokens: registers,
            num_registers,
            hidden_size,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        if image.rank() != 4 {
            return Err(SynaptixError::Unsupported("DinoV2: image must be [B, C, H, W]"));
        }
        let batch = image.dims()[0];
        let patches = self.vit.forward(image)?;
        let h = patches.dims()[patches.rank() - 1];

        let cls = self.cls_token.tensor()
            .reshape(vec![1, 1, h])?
            .broadcast_mul(&Tensor::ones(vec![batch, 1, h], patches.dtype(), patches.device())?)?;

        let prefix = if let Some(reg) = self.register_tokens.as_ref() {
            let reg_b = reg.tensor()
                .reshape(vec![1, self.num_registers, h])?
                .broadcast_mul(&Tensor::ones(vec![batch, self.num_registers, h], patches.dtype(), patches.device())?)?;
            Tensor::cat(&[&cls, &reg_b], 1)?
        } else {
            cls
        };
        let full = Tensor::cat(&[&prefix, &patches], 1)?;
        cls_pool(&full)
    }

    pub fn forward_features(&self, image: &Tensor) -> Result<Tensor> {
        self.vit.forward(image)
    }
}

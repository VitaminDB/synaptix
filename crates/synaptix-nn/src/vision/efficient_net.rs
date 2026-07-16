use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::conv::conv2d::conv2d;
use synaptix_ops::norm::group_norm::group_norm;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// EfficientNet (Tan & Le 2019) — compound-scaled CNN с MBConv-блоками.
///
/// **MBConvBlock**: `expand 1×1 → depthwise 3×3 → SE-block → project 1×1 →
/// +skip` (если stride=1 и channel match). SiLU вместо ReLU.
///
/// **Stem**: conv3×3 stride=2 → norm → silu. **Head**: conv1×1 (expand) →
/// silu → global avg pool → Linear(C, num_classes).
///
/// Реальная torchvision-реализация использует BatchNorm + DropPath; здесь
/// упрощения для inference (GroupNorm, без DropPath). Pretrained веса —
/// Phase O.
pub struct EfficientNet {
    pub stem_conv: Parameter,
    pub stem_norm_w: Parameter,
    pub stem_norm_b: Parameter,
    pub blocks: Vec<MbConvBlock>,
    pub head_conv: Parameter,
    pub head_norm_w: Parameter,
    pub head_norm_b: Parameter,
    pub classifier: Linear,
    pub in_channels: usize,
    pub num_classes: usize,
    pub norm_groups: usize,
}

pub struct MbConvBlock {
    pub expand_conv: Option<Parameter>,
    pub expand_norm_w: Option<Parameter>,
    pub expand_norm_b: Option<Parameter>,
    pub depthwise_conv: Parameter,
    pub depthwise_norm_w: Parameter,
    pub depthwise_norm_b: Parameter,
    pub project_conv: Parameter,
    pub project_norm_w: Parameter,
    pub project_norm_b: Parameter,
    pub stride: usize,
    pub use_residual: bool,
    pub norm_groups: usize,
}

impl EfficientNet {
    pub fn new(
        in_channels: usize, num_classes: usize,
        stem_out: usize, head_in: usize, head_out: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let stem_conv = crate::init::init_tensor(
            &[stem_out, in_channels, 3, 3],
            InitMethod::KaimingUniform { fan_in: in_channels * 9, a: 0.0 },
            dtype, 0, device,
        )?;
        let head_conv = crate::init::init_tensor(
            &[head_out, head_in, 1, 1],
            InitMethod::KaimingUniform { fan_in: head_in, a: 0.0 },
            dtype, 1, device,
        )?;
        let norm_groups = 8.min(stem_out.max(1));
        Ok(Self {
            stem_conv: Parameter::new(stem_conv),
            stem_norm_w: Parameter::new(Tensor::ones(vec![stem_out], dtype, device)?),
            stem_norm_b: Parameter::new(Tensor::zeros(vec![stem_out], dtype, device)?),
            blocks: Vec::new(),
            head_conv: Parameter::new(head_conv),
            head_norm_w: Parameter::new(Tensor::ones(vec![head_out], dtype, device)?),
            head_norm_b: Parameter::new(Tensor::zeros(vec![head_out], dtype, device)?),
            classifier: Linear::from_init(
                head_out, num_classes, true,
                InitMethod::XavierUniform { fan_in: head_out, fan_out: num_classes },
                InitMethod::Zeros, device, dtype, 2,
            )?,
            in_channels,
            num_classes,
            norm_groups,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        if image.rank() != 4 {
            return Err(SynaptixError::Unsupported("EfficientNet: image must be [B, C, H, W]"));
        }
        let mut h = conv2d(image, &self.stem_conv.tensor(), None, (2, 2), (1, 1), (1, 1))?;
        h = group_norm(
            &h, Some(&self.stem_norm_w.tensor()), Some(&self.stem_norm_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        h = h.silu()?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        h = conv2d(&h, &self.head_conv.tensor(), None, (1, 1), (0, 0), (1, 1))?;
        h = group_norm(
            &h, Some(&self.head_norm_w.tensor()), Some(&self.head_norm_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        h = h.silu()?;
        let dims = h.dims();
        let (b, c, hh, ww) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = h.reshape(vec![b, c, hh * ww])?.mean_keepdim(2)?.squeeze(2)?;
        self.classifier.forward(&flat)
    }
}

impl MbConvBlock {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let identity = x.clone();
        let mut h = if let Some(w) = self.expand_conv.as_ref() {
            let conv = conv2d(x, &w.tensor(), None, (1, 1), (0, 0), (1, 1))?;
            let normed = group_norm(
                &conv,
                self.expand_norm_w.as_ref().map(|p| p.tensor()).as_ref(),
                self.expand_norm_b.as_ref().map(|p| p.tensor()).as_ref(),
                self.norm_groups, 1e-5,
            )?;
            normed.silu()?
        } else {
            x.clone()
        };
        // Depthwise: упрощённо через conv2d с groups=1 (полный depthwise
        // через groups=C требует расширения conv2d API — Phase O).
        h = conv2d(&h, &self.depthwise_conv.tensor(), None, (self.stride, self.stride), (1, 1), (1, 1))?;
        h = group_norm(
            &h, Some(&self.depthwise_norm_w.tensor()), Some(&self.depthwise_norm_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        h = h.silu()?;
        h = conv2d(&h, &self.project_conv.tensor(), None, (1, 1), (0, 0), (1, 1))?;
        h = group_norm(
            &h, Some(&self.project_norm_w.tensor()), Some(&self.project_norm_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        if self.use_residual && self.stride == 1 && h.dims() == identity.dims() {
            h.add(&identity)
        } else {
            Ok(h)
        }
    }
}

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

/// ResNet (He et al. 2015) — CNN backbone с residual-блоками.
///
/// **BasicBlock** (для ResNet-18/34): `conv3x3 → norm → relu → conv3x3 →
/// norm → +skip → relu`. Используем GroupNorm вместо BatchNorm (batch-norm
/// требует running stats, GroupNorm — stateless и совместим с inference).
///
/// **Stem**: conv7x7 stride=2 → norm → relu → maxpool(заменён на stride=2
/// в первом stage для упрощения). Затем 4 stage'а с downsample-blocks.
/// **Head**: global avg pool → Linear(C, num_classes).
///
/// Реальная torchvision-реализация использует BatchNorm и MaxPool; здесь
/// упрощения для inference-only flow. Точное совпадение с pretrained
/// torchvision-весами требует BN — Phase O.
pub struct ResNet {
    pub stem_conv: Parameter,
    pub stem_norm_w: Parameter,
    pub stem_norm_b: Parameter,
    pub stages: Vec<ResNetStage>,
    pub head: Linear,
    pub in_channels: usize,
    pub num_classes: usize,
    pub norm_groups: usize,
}

pub struct ResNetStage {
    pub blocks: Vec<ResNetBasicBlock>,
    pub downsample: Option<ResNetDownsample>,
}

pub struct ResNetBasicBlock {
    pub conv1: Parameter,
    pub norm1_w: Parameter,
    pub norm1_b: Parameter,
    pub conv2: Parameter,
    pub norm2_w: Parameter,
    pub norm2_b: Parameter,
    pub stride: usize,
    pub norm_groups: usize,
}

pub struct ResNetDownsample {
    pub conv: Parameter,
    pub norm_w: Parameter,
    pub norm_b: Parameter,
}

impl ResNet {
    pub fn new(
        in_channels: usize, num_classes: usize,
        stem_out: usize, head_in: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let stem_conv = crate::init::init_tensor(
            &[stem_out, in_channels, 7, 7],
            InitMethod::KaimingUniform { fan_in: in_channels * 49, a: 0.0 },
            dtype, 0, device,
        )?;
        let norm_groups = 8.min(stem_out.max(1));
        Ok(Self {
            stem_conv: Parameter::new(stem_conv),
            stem_norm_w: Parameter::new(Tensor::ones(vec![stem_out], dtype, device)?),
            stem_norm_b: Parameter::new(Tensor::zeros(vec![stem_out], dtype, device)?),
            stages: Vec::new(),
            head: Linear::from_init(
                head_in, num_classes, true,
                InitMethod::XavierUniform { fan_in: head_in, fan_out: num_classes },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            in_channels,
            num_classes,
            norm_groups,
        })
    }

    pub fn forward(&self, image: &Tensor) -> Result<Tensor> {
        if image.rank() != 4 {
            return Err(SynaptixError::Unsupported("ResNet: image must be [B, C, H, W]"));
        }
        let mut h = conv2d(image, &self.stem_conv.tensor(), None, (2, 2), (3, 3), (1, 1))?;
        h = group_norm(
            &h, Some(&self.stem_norm_w.tensor()), Some(&self.stem_norm_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        h = h.relu()?;
        for stage in &self.stages {
            for (i, block) in stage.blocks.iter().enumerate() {
                let downsample = if i == 0 { stage.downsample.as_ref() } else { None };
                h = block.forward(&h, downsample)?;
            }
        }
        let dims = h.dims();
        let (b, c, hh, ww) = (dims[0], dims[1], dims[2], dims[3]);
        let flat = h.reshape(vec![b, c, hh * ww])?.mean_keepdim(2)?.squeeze(2)?;
        self.head.forward(&flat)
    }
}

impl ResNetBasicBlock {
    pub fn forward(&self, x: &Tensor, downsample: Option<&ResNetDownsample>) -> Result<Tensor> {
        let identity = match downsample {
            Some(ds) => {
                let d = conv2d(x, &ds.conv.tensor(), None, (self.stride, self.stride), (0, 0), (1, 1))?;
                group_norm(
                    &d, Some(&ds.norm_w.tensor()), Some(&ds.norm_b.tensor()),
                    self.norm_groups, 1e-5,
                )?
            }
            None => x.clone(),
        };
        let mut h = conv2d(x, &self.conv1.tensor(), None, (self.stride, self.stride), (1, 1), (1, 1))?;
        h = group_norm(
            &h, Some(&self.norm1_w.tensor()), Some(&self.norm1_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        h = h.relu()?;
        h = conv2d(&h, &self.conv2.tensor(), None, (1, 1), (1, 1), (1, 1))?;
        h = group_norm(
            &h, Some(&self.norm2_w.tensor()), Some(&self.norm2_b.tensor()),
            self.norm_groups, 1e-5,
        )?;
        h.add(&identity)?.relu()
    }
}

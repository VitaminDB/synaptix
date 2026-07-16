use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct SegmentationHead {
    pub proj: Linear,
    pub num_classes: usize,
}

impl SegmentationHead {
    pub fn new(hidden_size: usize, num_classes: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, num_classes, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            num_classes,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        let num_classes = proj.out_features();
        Ok(Self { proj, num_classes })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }

    pub fn forward_bchw(&self, x: &Tensor) -> Result<Tensor> {
        if x.rank() != 4 {
            return self.proj.forward(x);
        }
        let dims = x.dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let x_bhwc = x.permute(&[0, 2, 3, 1])?.contiguous()?;
        let x_flat = x_bhwc.reshape(&[b * h * w, c])?;
        let logits_flat = self.proj.forward(&x_flat)?;
        let logits_bhwc = logits_flat.reshape(&[b, h, w, self.num_classes])?;
        logits_bhwc.permute(&[0, 3, 1, 2])?.contiguous()
    }
}

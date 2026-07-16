use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct ClassificationHead {
    pub dense: Linear,
    pub out: Linear,
    pub num_classes: usize,
}

impl ClassificationHead {
    pub fn new(hidden_size: usize, num_classes: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            dense: Linear::from_init(
                hidden_size,
                hidden_size,
                true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros,
                device,
                dtype,
                0,
            )?,
            out: Linear::from_init(
                hidden_size,
                num_classes,
                true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros,
                device,
                dtype,
                1,
            )?,
            num_classes,
        })
    }

    pub fn from_weights(
        dense_w: Tensor, dense_b: Option<Tensor>,
        out_w: Tensor, out_b: Option<Tensor>,
    ) -> Result<Self> {
        let dense = Linear::new(dense_w, dense_b)?;
        let out_layer = Linear::new(out_w, out_b)?;
        let num_classes = out_layer.out_features();
        Ok(Self { dense, out: out_layer, num_classes })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.dense.forward(x)?;
        let activated = h.tanh()?;
        self.out.forward(&activated)
    }
}

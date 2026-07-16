use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct BboxHead {
    pub proj: Linear,
    pub num_classes: usize,
    pub sigmoid_output: bool,
}

impl BboxHead {
    pub fn new(hidden_size: usize, num_classes: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, num_classes * 4, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            num_classes,
            sigmoid_output: true,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, num_classes: usize, sigmoid_output: bool) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        if proj.out_features() != num_classes * 4 {
            return Err(synaptix_core::error::SynaptixError::shape_mismatch(
                &[num_classes * 4],
                &[proj.out_features()],
            ));
        }
        Ok(Self { proj, num_classes, sigmoid_output })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.proj.forward(x)?;
        let coords = if self.sigmoid_output { logits.sigmoid()? } else { logits };
        let mut new_dims: Vec<usize> = coords.dims().to_vec();
        let last = new_dims.pop().unwrap();
        debug_assert_eq!(last, self.num_classes * 4);
        new_dims.push(self.num_classes);
        new_dims.push(4);
        coords.reshape(new_dims)
    }
}

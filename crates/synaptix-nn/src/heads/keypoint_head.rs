use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

pub struct KeypointHead {
    pub proj: Linear,
    pub num_keypoints: usize,
    pub sigmoid_visibility: bool,
}

impl KeypointHead {
    pub fn new(hidden_size: usize, num_keypoints: usize, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            proj: Linear::from_init(hidden_size, num_keypoints * 3, true, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0)?,
            num_keypoints,
            sigmoid_visibility: true,
        })
    }

    pub fn from_weights(weight: Tensor, bias: Option<Tensor>, num_keypoints: usize, sigmoid_visibility: bool) -> Result<Self> {
        let proj = Linear::new(weight, bias)?;
        if proj.out_features() != num_keypoints * 3 {
            return Err(synaptix_core::error::SynaptixError::shape_mismatch(
                &[num_keypoints * 3],
                &[proj.out_features()],
            ));
        }
        Ok(Self { proj, num_keypoints, sigmoid_visibility })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let logits = self.proj.forward(x)?;
        let mut new_dims: Vec<usize> = logits.dims().to_vec();
        let last = new_dims.pop().unwrap();
        debug_assert_eq!(last, self.num_keypoints * 3);
        new_dims.push(self.num_keypoints);
        new_dims.push(3);
        let reshaped = logits.reshape(new_dims)?;
        if !self.sigmoid_visibility {
            return Ok(reshaped);
        }
        let kp_dim = reshaped.rank() - 1;
        let xy = reshaped.narrow(kp_dim, 0, 2)?.contiguous()?;
        let vis = reshaped.narrow(kp_dim, 2, 1)?.contiguous()?.sigmoid()?;
        Tensor::cat(&[&xy, &vis], kp_dim)?.contiguous()
    }
}

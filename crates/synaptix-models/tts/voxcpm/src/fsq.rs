use synaptix_core::tensor::Tensor;

use crate::loader::{Lin, VoxCheckpoint};
use crate::VoxError;

pub struct Fsq {
    in_proj: Lin,
    out_proj: Lin,
    scale: f32,
}

impl Fsq {
    pub fn load(ck: &VoxCheckpoint) -> Result<Self, VoxError> {
        Ok(Self {
            in_proj: Lin::load(ck, "fsq_layer", "in_proj", true)?,
            out_proj: Lin::load(ck, "fsq_layer", "out_proj", true)?,
            scale: ck.config.scalar_quantization_scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor, VoxError> {
        let h = self.in_proj.forward(x)?.tanh()?;
        let q = h.mul_scalar(self.scale)?.round()?.mul_scalar(1.0 / self.scale)?;
        self.out_proj.forward(&q)
    }
}

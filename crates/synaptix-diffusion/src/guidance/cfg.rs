use synaptix_core::tensor::Tensor;
use crate::error::{DiffusionError, Result};
use crate::guidance::Guidance;

pub struct Cfg {
    pub scale: f32,
}

impl Cfg {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl Guidance for Cfg {
    fn prepare_latents(&self, latent: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[latent, latent], 0).map_err(DiffusionError::from)
    }

    fn apply(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        uncond.add(&diff.affine(scale, 0.0)?).map_err(DiffusionError::from)
    }
}

pub fn apply_cfg(uncond: &Tensor, cond: &Tensor, scale: f32) -> Result<Tensor> {
    let diff = cond.sub(uncond)?;
    uncond.add(&diff.affine(scale, 0.0)?).map_err(DiffusionError::from)
}

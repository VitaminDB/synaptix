use synaptix_core::tensor::Tensor;
use crate::error::{DiffusionError, Result};
use crate::guidance::Guidance;

pub struct Pag {
    pub scale: f32,
    pub pag_applied_layers: Vec<String>,
}

impl Pag {
    pub fn new(scale: f32, layers: Vec<String>) -> Self {
        Self { scale, pag_applied_layers: layers }
    }
}

impl Guidance for Pag {
    fn prepare_latents(&self, latent: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[latent, latent, latent], 0).map_err(DiffusionError::from)
    }

    fn apply(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        uncond.add(&diff.affine(scale, 0.0)?).map_err(DiffusionError::from)
    }
}

pub fn apply_pag(uncond: &Tensor, perturbed: &Tensor, cond: &Tensor, cfg_scale: f32, pag_scale: f32) -> Result<Tensor> {
    let cfg_part = cond.sub(uncond)?.affine(cfg_scale, 0.0)?;
    let pag_part = cond.sub(perturbed)?.affine(pag_scale, 0.0)?;
    uncond.add(&cfg_part)?.add(&pag_part).map_err(DiffusionError::from)
}

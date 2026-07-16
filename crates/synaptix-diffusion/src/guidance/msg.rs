use synaptix_core::tensor::Tensor;
use crate::error::{DiffusionError, Result};
use crate::guidance::Guidance;

pub struct Msg {
    pub scale: f32,
    pub beta: f32,
    momentum: Option<Tensor>,
}

impl Msg {
    pub fn new(scale: f32, beta: f32) -> Self {
        Self { scale, beta, momentum: None }
    }

    pub fn reset(&mut self) {
        self.momentum = None;
    }

    pub fn step_guidance(&mut self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        let m = match &self.momentum {
            None => diff.clone(),
            Some(m) => m.affine(self.beta, 0.0)?.add(&diff.affine(1.0 - self.beta, 0.0)?)?,
        };
        self.momentum = Some(m.clone());
        uncond.add(&m.affine(scale, 0.0)?).map_err(DiffusionError::from)
    }
}

impl Guidance for Msg {
    fn prepare_latents(&self, latent: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[latent, latent], 0).map_err(DiffusionError::from)
    }

    fn apply(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        let m = match &self.momentum {
            None => diff.clone(),
            Some(m) => m.affine(self.beta, 0.0)?.add(&diff.affine(1.0 - self.beta, 0.0)?)?,
        };
        uncond.add(&m.affine(scale, 0.0)?).map_err(DiffusionError::from)
    }
}

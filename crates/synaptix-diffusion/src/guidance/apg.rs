use synaptix_core::tensor::Tensor;
use crate::error::{DiffusionError, Result};
use crate::guidance::Guidance;

pub struct Apg {
    pub scale: f32,
    pub momentum: f32,
    running_momentum: Option<Tensor>,
}

impl Apg {
    pub fn new(scale: f32, momentum: f32) -> Self {
        Self { scale, momentum, running_momentum: None }
    }

    pub fn reset(&mut self) {
        self.running_momentum = None;
    }
}

impl Guidance for Apg {
    fn prepare_latents(&self, latent: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[latent, latent], 0).map_err(DiffusionError::from)
    }

    fn apply(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor> {
        let diff = cond.sub(uncond)?;
        let flat = diff.flatten_all()?;
        let data: Vec<f32> = flat.to_vec1()?;
        let norm = data.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
        let proj_dir = diff.affine(1.0 / norm, 0.0)?;
        let proj_data: Vec<f32> = proj_dir.flatten_all()?.to_vec1()?;
        let cond_data: Vec<f32> = cond.flatten_all()?.to_vec1()?;
        let dot: f32 = cond_data.iter().zip(&proj_data).map(|(a, b)| a * b).sum();
        let cond_orth_data: Vec<f32> = cond_data.iter().zip(&proj_data)
            .map(|(c, p)| c - dot * p)
            .collect();
        let cond_orth = Tensor::from_vec(cond_orth_data, cond.dims().to_vec(), cond.device())?;
        let guided = cond_orth.add(&diff.affine(scale, 0.0)?)?;
        Ok(guided)
    }
}

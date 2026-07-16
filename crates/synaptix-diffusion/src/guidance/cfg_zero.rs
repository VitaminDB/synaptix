use synaptix_core::tensor::Tensor;
use crate::error::{DiffusionError, Result};
use crate::guidance::Guidance;

pub struct CfgZero {
    pub scale: f32,
}

impl CfgZero {
    pub fn new(scale: f32) -> Self {
        Self { scale }
    }
}

impl Guidance for CfgZero {
    fn prepare_latents(&self, latent: &Tensor) -> Result<Tensor> {
        Tensor::cat(&[latent, latent], 0).map_err(DiffusionError::from)
    }

    fn apply(&self, cond: &Tensor, uncond: &Tensor, scale: f32) -> Result<Tensor> {
        let cond_flat = cond.flatten_all()?;
        let uncond_flat = uncond.flatten_all()?;
        let cond_data: Vec<f32> = cond_flat.to_vec1()?;
        let uncond_data: Vec<f32> = uncond_flat.to_vec1()?;
        let dot_cu: f32 = cond_data.iter().zip(&uncond_data).map(|(a, b)| a * b).sum();
        let norm_u_sq: f32 = uncond_data.iter().map(|v| v * v).sum::<f32>().max(1e-12);
        let proj_scale = dot_cu / norm_u_sq;
        let uncond_proj = uncond.affine(proj_scale, 0.0)?;
        let cond_orth = cond.sub(&uncond_proj)?;
        let diff = cond_orth.add(uncond)?;
        uncond.add(&diff.affine(scale - 1.0, 0.0)?).map_err(DiffusionError::from)
    }
}

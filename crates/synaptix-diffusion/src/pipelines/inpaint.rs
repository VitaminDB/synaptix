use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{Scheduler, add_noise_ve, randn_like, cast_tensor};
use super::{DenoiserFn, PipelineOutput};

#[derive(Debug, Clone)]
pub struct InpaintConfig {
    pub n_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
    pub blend_masked: bool,
}

impl Default for InpaintConfig {
    fn default() -> Self {
        Self {
            n_steps: 50,
            guidance_scale: 7.5,
            seed: 0,
            blend_masked: true,
        }
    }
}

pub struct InpaintPipeline<'a> {
    pub config: InpaintConfig,
    pub scheduler: &'a mut dyn Scheduler,
}

impl<'a> InpaintPipeline<'a> {
    pub fn new(config: InpaintConfig, scheduler: &'a mut dyn Scheduler) -> Self {
        Self { config, scheduler }
    }

    pub fn run(
        &mut self,
        image_latents: Tensor,
        mask: Tensor,
        dtype: synaptix_core::dtype::DType,
        denoiser: &mut DenoiserFn,
        mut callback: Option<&mut dyn FnMut(usize, &Tensor)>,
    ) -> Result<PipelineOutput> {
        self.scheduler.set_timesteps(self.config.n_steps)?;

        let image_latents = cast_tensor(&image_latents, dtype)?;
        let mask = cast_tensor(&mask, dtype)?;

        let mut rng = Philox4x32::new(self.config.seed);
        let noise = randn_like(&image_latents, &mut rng)?;

        let init_sigma = self.scheduler.init_noise_sigma();
        let mut sample = add_noise_ve(&image_latents, &noise, init_sigma)?;

        let n = self.scheduler.n_steps();
        let sigmas = self.scheduler.sigmas().to_vec();

        for i in 0..n {
            let sigma = sigmas.get(i).copied().unwrap_or(0.0);
            let scaled = self.scheduler.scale_model_input(&sample, i)?;
            let model_out = denoiser(&scaled, sigma)?;
            let out = self.scheduler.step(&model_out, i, &sample)?;
            let mut next_sample = out.prev_sample;

            if self.config.blend_masked {
                let sigma_next = sigmas.get(i + 1).copied().unwrap_or(0.0);
                let mut rng2 = Philox4x32::new(self.config.seed.wrapping_add(i as u64 + 1));
                let fresh_noise = randn_like(&image_latents, &mut rng2)?;
                let noised_orig = add_noise_ve(&image_latents, &fresh_noise, sigma_next)?;
                let mask_inv = invert_mask(&mask)?;
                next_sample = blend_by_mask(&next_sample, &noised_orig, &mask_inv)?;
            }

            sample = next_sample;
            if let Some(cb) = callback.as_deref_mut() {
                cb(i, &sample);
            }
        }

        Ok(PipelineOutput::from_latents(sample))
    }
}

fn invert_mask(mask: &Tensor) -> Result<Tensor> {
    let ones = mask.affine(0.0, 1.0).map_err(DiffusionError::from)?;
    ones.sub(mask).map_err(DiffusionError::from)
}

fn blend_by_mask(a: &Tensor, b: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let a_part = a.mul(mask).map_err(DiffusionError::from)?;
    let mask_inv = invert_mask(mask)?;
    let b_part = b.mul(&mask_inv).map_err(DiffusionError::from)?;
    a_part.add(&b_part).map_err(DiffusionError::from)
}

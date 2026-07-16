use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{Scheduler, randn_seeded, cast_tensor};
use crate::conditioning::controlnet_cond::{ControlNetInput, ControlNetResiduals};
use super::{PipelineOutput};

#[derive(Debug, Clone)]
pub struct ControlNetConfig {
    pub height: usize,
    pub width: usize,
    pub latent_channels: usize,
    pub vae_scale_factor: usize,
    pub n_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for ControlNetConfig {
    fn default() -> Self {
        Self {
            height: 512,
            width: 512,
            latent_channels: 4,
            vae_scale_factor: 8,
            n_steps: 20,
            guidance_scale: 7.5,
            seed: 0,
        }
    }
}

impl ControlNetConfig {
    pub fn latent_height(&self) -> usize {
        self.height / self.vae_scale_factor
    }

    pub fn latent_width(&self) -> usize {
        self.width / self.vae_scale_factor
    }
}

pub struct ControlNetPipeline<'a> {
    pub config: ControlNetConfig,
    pub scheduler: &'a mut dyn Scheduler,
}

impl<'a> ControlNetPipeline<'a> {
    pub fn new(config: ControlNetConfig, scheduler: &'a mut dyn Scheduler) -> Self {
        Self { config, scheduler }
    }

    pub fn run(
        &mut self,
        control_input: ControlNetInput,
        device: synaptix_core::device::Device,
        dtype: synaptix_core::dtype::DType,
        mut controlnet_fn: impl FnMut(&Tensor, &ControlNetInput, f32) -> Result<ControlNetResiduals>,
        mut unet_fn: impl FnMut(&Tensor, &ControlNetResiduals, f32) -> Result<Tensor>,
        mut callback: Option<&mut dyn FnMut(usize, &Tensor)>,
    ) -> Result<PipelineOutput> {
        self.scheduler.set_timesteps(self.config.n_steps)?;
        let cfg = &self.config;
        let shape = [1, cfg.latent_channels, cfg.latent_height(), cfg.latent_width()];
        let mut rng = Philox4x32::new(cfg.seed);
        let noise = randn_seeded(&shape, device, &mut rng)?;
        let noise = cast_tensor(&noise, dtype)?;
        let init_sigma = self.scheduler.init_noise_sigma();
        let mut sample = noise.affine(init_sigma, 0.0).map_err(DiffusionError::from)?;

        let n = self.scheduler.n_steps();
        let sigmas = self.scheduler.sigmas().to_vec();

        for i in 0..n {
            let sigma = sigmas.get(i).copied().unwrap_or(0.0);
            let scaled = self.scheduler.scale_model_input(&sample, i)?;
            let residuals = controlnet_fn(&scaled, &control_input, sigma)?;
            let model_out = unet_fn(&scaled, &residuals, sigma)?;
            let out = self.scheduler.step(&model_out, i, &sample)?;
            sample = out.prev_sample;
            if let Some(cb) = callback.as_deref_mut() {
                cb(i, &sample);
            }
        }

        Ok(PipelineOutput::from_latents(sample))
    }
}

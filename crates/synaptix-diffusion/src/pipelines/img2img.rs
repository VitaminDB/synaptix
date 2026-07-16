use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::Result;
use crate::schedulers::{Scheduler, add_noise_ve, randn_like, cast_tensor};
use super::{DenoiserFn, PipelineOutput};

#[derive(Debug, Clone)]
pub struct Img2ImgConfig {
    pub n_steps: usize,
    pub strength: f32,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for Img2ImgConfig {
    fn default() -> Self {
        Self {
            n_steps: 50,
            strength: 0.8,
            guidance_scale: 7.5,
            seed: 0,
        }
    }
}

pub struct Img2ImgPipeline<'a> {
    pub config: Img2ImgConfig,
    pub scheduler: &'a mut dyn Scheduler,
}

impl<'a> Img2ImgPipeline<'a> {
    pub fn new(config: Img2ImgConfig, scheduler: &'a mut dyn Scheduler) -> Self {
        Self { config, scheduler }
    }

    pub fn run(
        &mut self,
        image_latents: Tensor,
        dtype: synaptix_core::dtype::DType,
        denoiser: &mut DenoiserFn,
        mut callback: Option<&mut dyn FnMut(usize, &Tensor)>,
    ) -> Result<PipelineOutput> {
        let strength = self.config.strength.clamp(0.0, 1.0);
        let total_steps = self.config.n_steps;
        let init_step = ((1.0 - strength) * total_steps as f32).round() as usize;

        self.scheduler.set_timesteps(total_steps)?;

        let sigmas = self.scheduler.sigmas().to_vec();
        let start_sigma = sigmas.get(init_step).copied().unwrap_or(*sigmas.first().unwrap_or(&1.0));

        let image_latents = cast_tensor(&image_latents, dtype)?;
        let mut rng = Philox4x32::new(self.config.seed);
        let noise = randn_like(&image_latents, &mut rng)?;
        let mut sample = add_noise_ve(&image_latents, &noise, start_sigma)?;

        let active_steps = total_steps.saturating_sub(init_step);
        for i in 0..active_steps {
            let global_i = init_step + i;
            let sigma = sigmas.get(global_i).copied().unwrap_or(0.0);
            let scaled = self.scheduler.scale_model_input(&sample, global_i)?;
            let model_out = denoiser(&scaled, sigma)?;
            let out = self.scheduler.step(&model_out, global_i, &sample)?;
            sample = out.prev_sample;
            if let Some(cb) = callback.as_deref_mut() {
                cb(i, &sample);
            }
        }

        Ok(PipelineOutput::from_latents(sample))
    }
}

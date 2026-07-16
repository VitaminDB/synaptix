use synaptix_core::tensor::Tensor;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{Scheduler, add_noise_ve, randn_like};
use crate::guidance::cfg::apply_cfg;
use super::{PipelineOutput};
use synaptix_ops::rng::Philox4x32;

#[derive(Debug, Clone)]
pub struct EditPipelineConfig {
    pub n_steps: usize,
    pub guidance_scale: f32,
    pub image_guidance_scale: f32,
    pub seed: u64,
}

impl Default for EditPipelineConfig {
    fn default() -> Self {
        Self {
            n_steps: 50,
            guidance_scale: 7.5,
            image_guidance_scale: 1.5,
            seed: 0,
        }
    }
}

pub struct EditPipeline<'a> {
    pub config: EditPipelineConfig,
    pub scheduler: &'a mut dyn Scheduler,
}

impl<'a> EditPipeline<'a> {
    pub fn new(config: EditPipelineConfig, scheduler: &'a mut dyn Scheduler) -> Self {
        Self { config, scheduler }
    }

    pub fn run(
        &mut self,
        image_latents: Tensor,
        dtype: synaptix_core::dtype::DType,
        mut denoiser_3way: impl FnMut(&Tensor, f32) -> Result<Tensor>,
        mut callback: Option<&mut dyn FnMut(usize, &Tensor)>,
    ) -> Result<PipelineOutput> {
        self.scheduler.set_timesteps(self.config.n_steps)?;

        let image_latents = image_latents.to_dtype(dtype).map_err(DiffusionError::from)?;
        let mut rng = Philox4x32::new(self.config.seed);
        let noise = randn_like(&image_latents, &mut rng)?;
        let init_sigma = self.scheduler.init_noise_sigma();
        let mut sample = add_noise_ve(&image_latents, &noise, init_sigma)?;

        let n = self.scheduler.n_steps();
        let sigmas = self.scheduler.sigmas().to_vec();
        let txt_scale = self.config.guidance_scale;
        let img_scale = self.config.image_guidance_scale;

        for i in 0..n {
            let sigma = sigmas.get(i).copied().unwrap_or(0.0);
            let scaled = self.scheduler.scale_model_input(&sample, i)?;
            let batch_out = denoiser_3way(&scaled, sigma)?;
            let model_out = apply_3way_cfg(&batch_out, txt_scale, img_scale)?;
            let out = self.scheduler.step(&model_out, i, &sample)?;
            sample = out.prev_sample;
            if let Some(cb) = callback.as_deref_mut() {
                cb(i, &sample);
            }
        }

        Ok(PipelineOutput::from_latents(sample))
    }
}

fn apply_3way_cfg(batch: &Tensor, txt_scale: f32, img_scale: f32) -> Result<Tensor> {
    let dims = batch.dims();
    if dims.is_empty() || dims[0] < 3 {
        return Err(DiffusionError::InvalidArgument(format!(
            "3-way CFG expects batch>=3, got {:?}",
            dims
        )));
    }
    let n = dims[0] / 3;
    let uncond = batch.narrow(0, 0, n).map_err(DiffusionError::from)?;
    let img_cond = batch.narrow(0, n, n).map_err(DiffusionError::from)?;
    let txt_cond = batch.narrow(0, 2 * n, dims[0] - 2 * n).map_err(DiffusionError::from)?;

    let after_img = apply_cfg(&uncond, &img_cond, img_scale)?;
    apply_cfg(&after_img, &txt_cond, txt_scale)
}

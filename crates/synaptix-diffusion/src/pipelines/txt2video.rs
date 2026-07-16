use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{Scheduler, randn_seeded, cast_tensor};
use crate::guidance::cfg::apply_cfg;
use super::{PipelineOutput};

#[derive(Debug, Clone)]
pub struct Txt2VideoConfig {
    pub num_frames: usize,
    pub height: usize,
    pub width: usize,
    pub latent_channels: usize,
    pub temporal_downsample: usize,
    pub vae_scale_factor: usize,
    pub n_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for Txt2VideoConfig {
    fn default() -> Self {
        Self {
            num_frames: 16,
            height: 256,
            width: 256,
            latent_channels: 4,
            temporal_downsample: 4,
            vae_scale_factor: 8,
            n_steps: 25,
            guidance_scale: 7.5,
            seed: 0,
        }
    }
}

impl Txt2VideoConfig {
    pub fn latent_frames(&self) -> usize {
        (self.num_frames + self.temporal_downsample - 1) / self.temporal_downsample
    }

    pub fn latent_height(&self) -> usize {
        self.height / self.vae_scale_factor
    }

    pub fn latent_width(&self) -> usize {
        self.width / self.vae_scale_factor
    }
}

pub struct Txt2VideoPipeline<'a> {
    pub config: Txt2VideoConfig,
    pub scheduler: &'a mut dyn Scheduler,
}

impl<'a> Txt2VideoPipeline<'a> {
    pub fn new(config: Txt2VideoConfig, scheduler: &'a mut dyn Scheduler) -> Self {
        Self { config, scheduler }
    }

    pub fn run(
        &mut self,
        device: synaptix_core::device::Device,
        dtype: synaptix_core::dtype::DType,
        guidance_scale: f32,
        mut cfg_denoiser: impl FnMut(&Tensor, f32) -> Result<Tensor>,
        mut callback: Option<&mut dyn FnMut(usize, &Tensor)>,
    ) -> Result<PipelineOutput> {
        self.scheduler.set_timesteps(self.config.n_steps)?;
        let cfg = &self.config;
        let shape = [
            1,
            cfg.latent_channels,
            cfg.latent_frames(),
            cfg.latent_height(),
            cfg.latent_width(),
        ];
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
            let batch_out = cfg_denoiser(&scaled, sigma)?;
            let (uncond, cond) = split_batch_2_video(&batch_out)?;
            let model_out = apply_cfg(&uncond, &cond, guidance_scale)?;
            let out = self.scheduler.step(&model_out, i, &sample)?;
            sample = out.prev_sample;
            if let Some(cb) = callback.as_deref_mut() {
                cb(i, &sample);
            }
        }

        Ok(PipelineOutput::from_latents(sample))
    }
}

fn split_batch_2_video(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = t.dims();
    if dims.is_empty() || dims[0] < 2 {
        return Err(DiffusionError::InvalidArgument(format!("batch={:?} < 2", dims.first())));
    }
    let b = dims[0];
    let half = b / 2;
    let uncond = t.narrow(0, 0, half).map_err(DiffusionError::from)?;
    let cond = t.narrow(0, half, b - half).map_err(DiffusionError::from)?;
    Ok((uncond, cond))
}

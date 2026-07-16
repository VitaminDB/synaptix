use synaptix_core::tensor::Tensor;
use synaptix_ops::rng::Philox4x32;

use crate::error::{DiffusionError, Result};
use crate::schedulers::{Scheduler, randn_seeded, cast_tensor};
use crate::guidance::cfg::apply_cfg;
use super::{PipelineOutput};

#[derive(Debug, Clone)]
pub struct Txt2AudioConfig {
    pub latent_seq_len: usize,
    pub latent_channels: usize,
    pub n_steps: usize,
    pub guidance_scale: f32,
    pub seed: u64,
}

impl Default for Txt2AudioConfig {
    fn default() -> Self {
        Self {
            latent_seq_len: 256,
            latent_channels: 64,
            n_steps: 60,
            guidance_scale: 5.0,
            seed: 0,
        }
    }
}

pub struct Txt2AudioPipeline<'a> {
    pub config: Txt2AudioConfig,
    pub scheduler: &'a mut dyn Scheduler,
}

impl<'a> Txt2AudioPipeline<'a> {
    pub fn new(config: Txt2AudioConfig, scheduler: &'a mut dyn Scheduler) -> Self {
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
        let shape = [1, cfg.latent_channels, cfg.latent_seq_len];
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
            let (uncond, cond) = split_audio_batch(&batch_out)?;
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

fn split_audio_batch(t: &Tensor) -> Result<(Tensor, Tensor)> {
    let dims = t.dims();
    if dims.is_empty() || dims[0] < 2 {
        return Err(DiffusionError::InvalidArgument(format!("audio batch={:?} < 2", dims.first())));
    }
    let b = dims[0];
    let half = b / 2;
    let uncond = t.narrow(0, 0, half).map_err(DiffusionError::from)?;
    let cond = t.narrow(0, half, b - half).map_err(DiffusionError::from)?;
    Ok((uncond, cond))
}

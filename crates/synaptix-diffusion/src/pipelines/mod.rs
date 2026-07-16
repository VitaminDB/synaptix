pub mod controlnet;
pub mod edit_pipeline;
pub mod img2img;
pub mod inpaint;
pub mod txt2audio;
pub mod txt2img;
pub mod txt2video;

use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub type DenoiserFn = Box<dyn FnMut(&Tensor, f32) -> Result<Tensor> + Send>;

#[derive(Debug, Clone)]
pub struct PipelineOutput {
    pub latents: Tensor,
    pub images: Option<Vec<Tensor>>,
}

impl PipelineOutput {
    pub fn from_latents(latents: Tensor) -> Self {
        Self { latents, images: None }
    }
}

pub fn run_denoising_loop(
    scheduler: &mut dyn crate::schedulers::Scheduler,
    latents: &Tensor,
    denoiser: &mut DenoiserFn,
    mut callback: Option<&mut dyn FnMut(usize, &Tensor)>,
) -> Result<Tensor> {
    let mut sample = latents.clone();
    let n = scheduler.n_steps();
    for i in 0..n {
        let scaled = scheduler.scale_model_input(&sample, i)?;
        let sigma = scheduler.sigmas().get(i).copied().unwrap_or(0.0);
        let model_out = denoiser(&scaled, sigma)?;
        let out = scheduler.step(&model_out, i, &sample)?;
        sample = out.prev_sample;
        if let Some(cb) = callback.as_deref_mut() {
            cb(i, &sample);
        }
    }
    Ok(sample)
}

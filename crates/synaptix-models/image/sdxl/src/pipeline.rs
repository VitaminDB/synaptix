//! txt2img-пайплайн: промпт → латентный шум → denoising-петля (CFG + Euler) →
//! VAE-декод → RGB-изображение.
//!
//! Планировщик — `EulerScheduler` с конфигом SDXL-base (scaled_linear betas
//! 0.00085→0.012, leading-spacing, epsilon-prediction) — это дефолтный
//! `EulerDiscreteScheduler` диффузерса для SDXL. На каждом шаге латент
//! дублируется в батч-2 (uncond, cond), один forward UNet, затем CFG:
//! `eps = eps_uncond + scale·(eps_cond − eps_uncond)`. Conditioning (две
//! башни CLIP + pooled bigG + `add_time_ids`) считается один раз до петли.

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_diffusion::apply_cfg;
use synaptix_diffusion::schedulers::euler::{EulerConfig, EulerScheduler};
use synaptix_diffusion::schedulers::{randn_seeded, Scheduler};
use synaptix_ops::rng::Philox4x32;

use crate::config::Txt2ImgParams;
use crate::model::SdxlModel;
use crate::SdxlError;

pub struct SdxlPipeline {
    pub model: SdxlModel,
}

impl SdxlPipeline {
    pub fn from_pretrained(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, SdxlError> {
        Ok(Self { model: SdxlModel::from_pretrained(dir, device, dtype)? })
    }

    /// Как [`from_pretrained`], с квантованием весов UNet (`quant` = NVFP4/MXFP8;
    /// иначе dense). См. [`SdxlModel::from_pretrained_quant`].
    pub fn from_pretrained_quant(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        quant: DType,
    ) -> Result<Self, SdxlError> {
        Ok(Self { model: SdxlModel::from_pretrained_quant(dir, device, dtype, quant)? })
    }

    /// SDXL `add_time_ids` = (orig_h, orig_w, crop_top, crop_left, target_h,
    /// target_w). Без кропа orig == target == запрошенный размер. Батч-2
    /// (одинаково для uncond и cond).
    fn time_ids(&self, h: usize, w: usize) -> Result<Tensor, SdxlError> {
        let (h, w) = (h as f32, w as f32);
        let row = [h, w, 0.0, 0.0, h, w];
        let data: Vec<f32> = row.iter().chain(row.iter()).copied().collect();
        Ok(Tensor::from_vec(data, (2, 6), self.model.device)?)
    }

    /// Полный txt2img. `callback(step, total)` зовётся после каждого шага
    /// (для прогресс-бара). Возвращает CHW-изображение `[3, H, W]` (F32, [0,1]).
    pub fn txt2img(
        &self,
        params: &Txt2ImgParams,
        mut callback: impl FnMut(usize, usize),
    ) -> Result<Tensor, SdxlError> {
        // Inference-only: gradient-трекинг выкл. Включает fused-пути, gated на
        // !is_grad_enabled (layer_norm_fused и пр.) + убирает grad-fn оверхед.
        let _ng = synaptix_core::grad::NoGradGuard::new();
        let dev = self.model.device;
        let dtype = self.model.dtype;

        let cond = self.model.encode_prompt(&params.prompt, &params.negative_prompt)?;
        let time_ids = self.time_ids(params.height, params.width)?;

        let mut scheduler = EulerScheduler::new(EulerConfig::default());
        scheduler.set_timesteps(params.steps)?;
        let timesteps: Vec<f32> = scheduler.timesteps().to_vec();
        let n = scheduler.n_steps();

        let shape = [1, 4, params.latent_height(), params.latent_width()];
        let mut rng = Philox4x32::new(params.seed);
        let noise = randn_seeded(&shape, dev, &mut rng)?.to_dtype(dtype)?;
        let mut latents = noise.affine(scheduler.init_noise_sigma(), 0.0)?;

        for i in 0..n {
            let scaled = scheduler.scale_model_input(&latents, i)?;
            let latent_in = Tensor::cat(&[&scaled, &scaled], 0)?.contiguous()?;
            let t = timesteps.get(i).copied().unwrap_or(0.0);
            let t_tensor = Tensor::from_vec(vec![t, t], (2,), dev)?;

            let noise_pred = self.model.unet.forward(
                &latent_in,
                &t_tensor,
                &cond.encoder_hidden_states,
                &cond.pooled,
                &time_ids,
            )?;

            let uncond = noise_pred.narrow(0, 0, 1)?;
            let cond_pred = noise_pred.narrow(0, 1, 1)?;
            let model_out = apply_cfg(&uncond, &cond_pred, params.guidance_scale)?;

            let out = scheduler.step(&model_out, i, &latents)?;
            latents = out.prev_sample;
            callback(i + 1, n);
        }

        self.decode_latents(&latents)
    }

    /// VAE-декод латента в RGB. `KlVae::decode` сам делит на `scaling_factor`,
    /// затем post_quant_conv + decoder. Выход в [-1,1] → денорм в [0,1] и срез
    /// батча. Латент приводится к `vae_dtype` (F32 или BF16); итог — F32.
    fn decode_latents(&self, latents: &Tensor) -> Result<Tensor, SdxlError> {
        let z = latents.to_dtype(self.model.vae_dtype)?;
        let image = self.model.vae.decode(&z)?;
        let image = image.affine(0.5, 0.5)?;
        let dims = image.dims().to_vec();
        let chw = image.narrow(0, 0, 1)?.reshape(vec![dims[1], dims[2], dims[3]])?;
        Ok(chw.contiguous()?.to_dtype(DType::F32)?)
    }
}

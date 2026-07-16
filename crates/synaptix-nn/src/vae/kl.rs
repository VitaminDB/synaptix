//! `KlVae` — полный `AutoencoderKL` (настоящий conv2d encoder + decoder) с
//! диагональным гауссовым латентом и `scaling_factor`. Совместим с diffusers
//! `AutoencoderKL` (SD-1.x / SDXL / SD3 — отличаются только конфигом).

use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::vae::autoencoder_kl::{AutoencoderKlConfig, AutoencoderKlDecoder, AutoencoderKlEncoder};

/// Полный `AutoencoderKL`: encoder → diagonal-Gaussian латент → decoder.
pub struct KlVae {
    pub encoder: AutoencoderKlEncoder,
    pub decoder: AutoencoderKlDecoder,
    pub scaling_factor: f32,
}

impl KlVae {
    /// Загрузка из источника весов по HF-именам diffusers (`encoder.*`,
    /// `decoder.*`, `quant_conv.*`, `post_quant_conv.*`).
    pub fn load<F>(cfg: &AutoencoderKlConfig, get: &F) -> Result<Self>
    where
        F: Fn(&str) -> Result<Tensor>,
    {
        Ok(Self {
            encoder: AutoencoderKlEncoder::load(cfg, get)?,
            decoder: AutoencoderKlDecoder::load(cfg, get)?,
            scaling_factor: cfg.scaling_factor,
        })
    }

    /// `x: [B, C, H, W]` → moments `[B, 2·latent, h, w]` (== diffusers
    /// `vae.encode(x).latent_dist.parameters`, после `quant_conv`).
    pub fn encode_moments(&self, x: &Tensor) -> Result<Tensor> {
        self.encoder.encode(x)
    }

    /// Сэмпл латента, масштабированный на `scaling_factor` (как diffusers
    /// `latents = vae.encode(x).latent_dist.sample() * scaling_factor`).
    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let moments = self.encoder.encode(x)?;
        let (mean, logvar) = self.encoder.split_moments(&moments)?;
        let z = reparameterize(&mean, &logvar)?;
        z.affine(self.scaling_factor, 0.0)
    }

    /// `z: [B, latent, h, w]` (масштабированный латент) → image. Делит на
    /// `scaling_factor`, затем `post_quant_conv` + decoder.
    pub fn decode(&self, z: &Tensor) -> Result<Tensor> {
        let z = z.affine(1.0 / self.scaling_factor, 0.0)?;
        self.decoder.decode(&z)
    }
}

pub fn reparameterize(mean: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    reparameterize_with_eps(mean, logvar, None)
}

pub fn reparameterize_with_eps(mean: &Tensor, logvar: &Tensor, eps: Option<&Tensor>) -> Result<Tensor> {
    let std = logvar.affine(0.5, 0.0)?.exp()?;
    let noise = match eps {
        Some(e) => e.clone(),
        None => Tensor::randn(mean.dims().to_vec(), mean.device())?.to_dtype(mean.dtype())?,
    };
    mean.add(&noise.mul(&std)?)
}

pub fn kl_divergence(mean: &Tensor, logvar: &Tensor) -> Result<Tensor> {
    let mean_sq = mean.mul(mean)?;
    let var = logvar.exp()?;
    let term = mean_sq.add(&var)?.sub(&logvar)?.affine(1.0, -1.0)?;
    term.affine(0.5, 0.0)
}

//! Вокодер: BigVGAN v2 22 кГц (`nvidia/bigvgan_v2_22khz_80band_256x`) — общий
//! [`synaptix_nn::audio::bigvgan`], веса из компонента `bigvgan`.

use synaptix_core::{dtype::DType, tensor::Tensor};
use synaptix_nn::audio::bigvgan::{BigVgan, BigVganConfig, Weights};

use crate::loader::Comp;
use crate::nn::host;
use crate::R;

pub struct Vocoder(BigVgan);

impl Vocoder {
    pub fn load(c: &Comp) -> R<Self> {
        let get = |name: &str| c.get(name, DType::F32).map_err(|e| synaptix_core::error::SynaptixError::Other(e.to_string()));
        let has = |name: &str| c.has(name);
        Ok(Self(BigVgan::load(&Weights { get: &get, has: &has }, &BigVganConfig::v2_22khz_80band_256x(), "", c.device(), DType::F32)?))
    }

    /// Мел `[1, 80, T]` → волна `T·256` сэмплов в [−1, 1].
    pub fn decode(&self, mel: &Tensor) -> R<Vec<f32>> {
        host(&self.0.forward(mel)?)
    }
}

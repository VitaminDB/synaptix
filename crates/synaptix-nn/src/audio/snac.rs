use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::audio::rvq::ResidualVQ;
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// SNAC — Multi-Scale Neural Audio Codec (Siuzdak et al. 2024).
///
/// **Multi-scale RVQ**: каждый scale работает на своём временном разрешении
/// (downsampled последовательность). Здесь — минимальный composite с N=`scales.len()`
/// независимыми RVQ-стеками; полный SNAC использует downsample-conv между
/// scales (откладывается до Phase O). `forward` суммирует reconstructions от всех scale'ов.
pub struct Snac {
    pub encoder_proj: Linear,
    pub decoder_proj: Linear,
    pub quantizers: Vec<ResidualVQ>,
    pub in_channels: usize,
    pub hidden_size: usize,
    pub scales: Vec<usize>,
}

impl Snac {
    pub fn new(
        in_channels: usize, hidden_size: usize,
        scales: Vec<usize>, codebook_size: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(scales.len());
        for &num_codebooks in scales.iter() {
            quantizers.push(ResidualVQ::new(
                num_codebooks, codebook_size, hidden_size, device, dtype,
            )?);
        }
        Ok(Self {
            encoder_proj: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            decoder_proj: Linear::from_init(
                hidden_size, in_channels, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            quantizers,
            in_channels,
            hidden_size,
            scales,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder_proj.forward(x)?;
        let mut acc: Option<Tensor> = None;
        for q in &self.quantizers {
            let r = q.forward(&h)?;
            acc = Some(match acc {
                None => r,
                Some(prev) => prev.add(&r)?,
            });
        }
        let combined = acc.unwrap_or(h);
        self.decoder_proj.forward(&combined)
    }
}

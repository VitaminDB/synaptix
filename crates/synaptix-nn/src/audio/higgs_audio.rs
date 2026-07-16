use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::audio::rvq::ResidualVQ;
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// Higgs Audio v2 tokenizer (Boson AI 2024).
///
/// Минимальный composite: encoder_proj → RVQ → decoder_proj. Параметры
/// аналогичны DAC/EnCodec, но Higgs Audio в исходной реализации использует
/// более узкие codebook'и (12-dim) + dropout-aware quantization — отложено
/// в Phase O.
pub struct HiggsAudio {
    pub encoder_proj: Linear,
    pub decoder_proj: Linear,
    pub quantizer: ResidualVQ,
    pub in_channels: usize,
    pub hidden_size: usize,
}

impl HiggsAudio {
    pub fn new(
        in_channels: usize, hidden_size: usize,
        num_codebooks: usize, codebook_size: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
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
            quantizer: ResidualVQ::new(num_codebooks, codebook_size, hidden_size, device, dtype)?,
            in_channels,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder_proj.forward(x)?;
        let q = self.quantizer.forward(&h)?;
        self.decoder_proj.forward(&q)
    }
}

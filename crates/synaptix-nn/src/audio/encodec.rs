use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::audio::rvq::ResidualVQ;
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// EnCodec — Meta high-fidelity audio codec (Défossez et al. 2022).
///
/// Минимальный composite (encoder_proj → RVQ → decoder_proj). Полный EnCodec
/// использует conv-encoder со снэк-активациями + LSTM + conv-decoder; здесь
/// только quantization-уровень. Дефолты: 8 codebook'ов × 1024 entries.
pub struct EnCodec {
    pub encoder_proj: Linear,
    pub decoder_proj: Linear,
    pub quantizer: ResidualVQ,
    pub in_channels: usize,
    pub hidden_size: usize,
}

impl EnCodec {
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

    pub fn encode(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder_proj.forward(x)?;
        self.quantizer.encode(&h)
    }

    pub fn decode(&self, indices: &Tensor, dtype: DType) -> Result<Tensor> {
        let recon = self.quantizer.decode(indices, dtype)?;
        self.decoder_proj.forward(&recon)
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder_proj.forward(x)?;
        let q = self.quantizer.forward(&h)?;
        self.decoder_proj.forward(&q)
    }
}

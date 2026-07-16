use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::audio::rvq::ResidualVQ;
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;

/// Mimi — Moshi audio codec (Kyutai 2024).
///
/// **Semantic + acoustic split**: первые `semantic_codebooks` кодбуков
/// дистиллированы из WavLM-семантической модели, остальные — acoustic.
/// Здесь минимальный composite с двумя независимыми RVQ-стеками
/// (semantic и acoustic), `forward` суммирует их reconstructions.
/// Полная семантическая дистилляция — откладывается до Phase O.
pub struct Mimi {
    pub encoder_proj: Linear,
    pub decoder_proj: Linear,
    pub semantic_quantizer: ResidualVQ,
    pub acoustic_quantizer: ResidualVQ,
    pub in_channels: usize,
    pub hidden_size: usize,
}

impl Mimi {
    pub fn new(
        in_channels: usize, hidden_size: usize,
        semantic_codebooks: usize, acoustic_codebooks: usize, codebook_size: usize,
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
            semantic_quantizer: ResidualVQ::new(
                semantic_codebooks, codebook_size, hidden_size, device, dtype,
            )?,
            acoustic_quantizer: ResidualVQ::new(
                acoustic_codebooks, codebook_size, hidden_size, device, dtype,
            )?,
            in_channels,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.encoder_proj.forward(x)?;
        let semantic = self.semantic_quantizer.forward(&h)?;
        let acoustic = self.acoustic_quantizer.forward(&h)?;
        let combined = semantic.add(&acoustic)?;
        self.decoder_proj.forward(&combined)
    }
}

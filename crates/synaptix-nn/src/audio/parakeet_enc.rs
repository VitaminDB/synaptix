use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;
use synaptix_ops::norm::layer_norm;

use crate::audio::conformer_enc::ConformerBlock;
use crate::init::InitMethod;
use crate::linear::Linear;
use crate::module::Module;
use crate::parameter::Parameter;

/// NVIDIA Parakeet RNN-T encoder (Conformer-XL стиль).
///
/// `in_proj → N×ConformerBlock → final_ln → out_proj`. Joint network
/// (RNN-T head) живёт отдельно в `synaptix-nn::heads::RnnTHead`.
pub struct ParakeetEnc {
    pub in_proj: Linear,
    pub blocks: Vec<ConformerBlock>,
    pub final_ln_w: Parameter,
    pub final_ln_b: Parameter,
    pub out_proj: Linear,
    pub in_channels: usize,
    pub hidden_size: usize,
}

impl ParakeetEnc {
    pub fn new(
        in_channels: usize, hidden_size: usize,
        device: Device, dtype: DType,
    ) -> Result<Self> {
        let ln_w = Tensor::ones(vec![hidden_size], dtype, device)?;
        let ln_b = Tensor::zeros(vec![hidden_size], dtype, device)?;
        Ok(Self {
            in_proj: Linear::from_init(
                in_channels, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: in_channels, a: 0.0 },
                InitMethod::Zeros, device, dtype, 0,
            )?,
            blocks: Vec::new(),
            final_ln_w: Parameter::new(ln_w),
            final_ln_b: Parameter::new(ln_b),
            out_proj: Linear::from_init(
                hidden_size, hidden_size, true,
                InitMethod::KaimingUniform { fan_in: hidden_size, a: 0.0 },
                InitMethod::Zeros, device, dtype, 1,
            )?,
            in_channels,
            hidden_size,
        })
    }

    pub fn from_parts(
        in_proj: Linear, blocks: Vec<ConformerBlock>,
        final_ln_w: Tensor, final_ln_b: Tensor, out_proj: Linear,
    ) -> Result<Self> {
        let in_channels = in_proj.in_features();
        let hidden_size = in_proj.out_features();
        Ok(Self {
            in_proj,
            blocks,
            final_ln_w: Parameter::new(final_ln_w),
            final_ln_b: Parameter::new(final_ln_b),
            out_proj,
            in_channels,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut h = self.in_proj.forward(x)?;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        let normed = layer_norm(&h, Some(&self.final_ln_w.tensor()), Some(&self.final_ln_b.tensor()), 1e-5)?;
        self.out_proj.forward(&normed)
    }
}

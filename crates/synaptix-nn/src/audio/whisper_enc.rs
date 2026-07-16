use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use synaptix_ops::activation::gelu_tanh;
use synaptix_ops::conv::conv1d;
use synaptix_ops::norm::layer_norm;

use crate::init::InitMethod;
use crate::parameter::Parameter;
use crate::transformer::TransformerBlock;

pub struct WhisperEnc {
    pub conv1_w: Parameter,
    pub conv1_b: Parameter,
    pub conv2_w: Parameter,
    pub conv2_b: Parameter,
    pub blocks: Vec<TransformerBlock>,
    pub final_norm_w: Parameter,
    pub final_norm_b: Parameter,
    pub hidden_size: usize,
}

impl WhisperEnc {
    pub fn from_weights(
        conv1_w: Tensor, conv1_b: Tensor,
        conv2_w: Tensor, conv2_b: Tensor,
        blocks: Vec<TransformerBlock>,
        final_norm_w: Tensor, final_norm_b: Tensor,
    ) -> Result<Self> {
        let hidden_size = conv2_w.dims()[0];
        Ok(Self {
            conv1_w: Parameter::new(conv1_w),
            conv1_b: Parameter::new(conv1_b),
            conv2_w: Parameter::new(conv2_w),
            conv2_b: Parameter::new(conv2_b),
            blocks,
            final_norm_w: Parameter::new(final_norm_w),
            final_norm_b: Parameter::new(final_norm_b),
            hidden_size,
        })
    }

    pub fn new(_in_channels: usize, hidden_size: usize, device: Device, dtype: DType) -> Result<Self> {
        let nw = crate::init::init_tensor(&[hidden_size], InitMethod::Ones, dtype, 0, device)?;
        let nb = crate::init::init_tensor(&[hidden_size], InitMethod::Zeros, dtype, 1, device)?;
        let dummy = crate::init::init_tensor(&[1, 1, 1], InitMethod::Zeros, dtype, 2, device)?;
        let dummy_b = crate::init::init_tensor(&[1], InitMethod::Zeros, dtype, 3, device)?;
        Ok(Self {
            conv1_w: Parameter::new(dummy.clone()),
            conv1_b: Parameter::new(dummy_b.clone()),
            conv2_w: Parameter::new(dummy),
            conv2_b: Parameter::new(dummy_b),
            blocks: Vec::new(),
            final_norm_w: Parameter::new(nw),
            final_norm_b: Parameter::new(nb),
            hidden_size,
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        if mel.rank() != 3 {
            return Err(SynaptixError::Unsupported("whisper_enc: mel must be [B, n_mels, T]"));
        }
        let x = conv1d(mel, &self.conv1_w.tensor(), Some(&self.conv1_b.tensor()), 1, 1)?;
        let x = gelu_tanh(&x)?;
        let x = conv1d(&x, &self.conv2_w.tensor(), Some(&self.conv2_b.tensor()), 2, 1)?;
        let x = gelu_tanh(&x)?;
        let x = x.transpose(1, 2)?.contiguous()?;
        let mut h = x;
        for block in &self.blocks {
            h = block.forward(&h)?;
        }
        layer_norm(&h, Some(&self.final_norm_w.tensor()), Some(&self.final_norm_b.tensor()), 1e-5)
    }
}

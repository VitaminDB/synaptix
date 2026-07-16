use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::transformer::decoder::TransformerDecoder;
use crate::transformer::encoder::TransformerEncoder;

pub struct EncoderDecoder {
    pub encoder: TransformerEncoder,
    pub decoder: TransformerDecoder,
}

impl EncoderDecoder {
    pub fn new(
        enc_layers: usize,
        dec_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        ffn_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        Ok(Self {
            encoder: TransformerEncoder::new(enc_layers, hidden_size, num_heads, ffn_dim, device, dtype)?,
            decoder: TransformerDecoder::new(dec_layers, hidden_size, num_heads, ffn_dim, device, dtype)?,
        })
    }

    pub fn forward(&self, src: &Tensor, tgt: &Tensor) -> Result<Tensor> {
        let enc_out = self.encoder.forward(src)?;
        self.decoder.forward_with_context(tgt, &enc_out, None)
    }
}

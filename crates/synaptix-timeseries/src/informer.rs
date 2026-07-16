use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct InformerConfig {
    pub seq_len: usize,
    pub label_len: usize,
    pub pred_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_encoder_layers: usize,
    pub num_decoder_layers: usize,
}

impl Default for InformerConfig {
    fn default() -> Self {
        Self { seq_len: 96, label_len: 48, pred_len: 24, d_model: 512, num_heads: 8, num_encoder_layers: 2, num_decoder_layers: 1 }
    }
}

pub struct Informer {
    pub config: InformerConfig,
}

impl Informer {
    pub fn new(config: InformerConfig) -> Self { Self { config } }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
}

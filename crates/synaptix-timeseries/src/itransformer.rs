use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct ItransformerConfig {
    pub seq_len: usize,
    pub pred_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

impl Default for ItransformerConfig {
    fn default() -> Self {
        Self { seq_len: 96, pred_len: 96, d_model: 512, num_heads: 8, num_layers: 3 }
    }
}

pub struct Itransformer {
    pub config: ItransformerConfig,
}

impl Itransformer {
    pub fn new(config: ItransformerConfig) -> Self { Self { config } }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
}

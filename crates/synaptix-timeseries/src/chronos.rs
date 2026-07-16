use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct ChronosConfig {
    pub context_len: usize,
    pub prediction_len: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub vocab_size: usize,
}

impl Default for ChronosConfig {
    fn default() -> Self {
        Self { context_len: 512, prediction_len: 64, d_model: 768, num_heads: 12, num_layers: 12, vocab_size: 4096 }
    }
}

pub struct Chronos {
    pub config: ChronosConfig,
}

impl Chronos {
    pub fn new(config: ChronosConfig) -> Self { Self { config } }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
}

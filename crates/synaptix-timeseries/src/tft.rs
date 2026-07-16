use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct TftConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub context_len: usize,
    pub horizon: usize,
}

impl Default for TftConfig {
    fn default() -> Self {
        Self { d_model: 160, num_heads: 4, num_layers: 3, context_len: 168, horizon: 24 }
    }
}

pub struct Tft {
    pub config: TftConfig,
}

impl Tft {
    pub fn new(config: TftConfig) -> Self { Self { config } }
    pub fn forward(&self, x: &Tensor, _future: Option<&Tensor>) -> Result<Tensor> { Ok(x.clone()) }
}

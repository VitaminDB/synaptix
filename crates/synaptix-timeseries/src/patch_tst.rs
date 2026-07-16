use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct PatchTstConfig {
    pub patch_len: usize,
    pub stride: usize,
    pub d_model: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub horizon: usize,
}

impl Default for PatchTstConfig {
    fn default() -> Self {
        Self { patch_len: 16, stride: 8, d_model: 128, num_heads: 16, num_layers: 3, horizon: 96 }
    }
}

pub struct PatchTst {
    pub config: PatchTstConfig,
}

impl PatchTst {
    pub fn new(config: PatchTstConfig) -> Self { Self { config } }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
}

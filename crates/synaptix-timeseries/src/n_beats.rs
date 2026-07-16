use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct NBeatsConfig {
    pub num_stacks: usize,
    pub num_blocks: usize,
    pub num_layers: usize,
    pub layer_width: usize,
    pub horizon: usize,
}

impl Default for NBeatsConfig {
    fn default() -> Self {
        Self { num_stacks: 2, num_blocks: 3, num_layers: 4, layer_width: 512, horizon: 24 }
    }
}

pub struct NBeats {
    pub config: NBeatsConfig,
}

impl NBeats {
    pub fn new(config: NBeatsConfig) -> Self { Self { config } }
    pub fn forecast(&self, x: &Tensor) -> Result<Tensor> { Ok(x.clone()) }
}

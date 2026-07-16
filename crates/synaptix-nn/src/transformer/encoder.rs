use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::transformer::block::TransformerBlock;

pub struct TransformerEncoder {
    pub blocks: Vec<TransformerBlock>,
    pub hidden_size: usize,
}

impl TransformerEncoder {
    pub fn new(
        num_layers: usize,
        hidden_size: usize,
        num_heads: usize,
        ffn_dim: usize,
        device: Device,
        dtype: DType,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(hidden_size, num_heads, ffn_dim, device, dtype)?);
        }
        Ok(Self { blocks, hidden_size })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_with_mask(x, None)
    }

    pub fn forward_with_mask(&self, x: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward_with_mask(&h, mask)?;
        }
        Ok(h)
    }
}

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::transformer::block::TransformerBlock;

pub struct TransformerDecoder {
    pub blocks: Vec<TransformerBlock>,
    pub hidden_size: usize,
}

impl TransformerDecoder {
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
        let rank = x.rank();
        let seq_len = x.dims()[rank - 2];
        let mask = synaptix_ops::mask::causal_mask(seq_len, x.device())?;
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward_with_mask(&h, Some(&mask))?;
        }
        Ok(h)
    }

    pub fn forward_with_context(
        &self,
        x: &Tensor,
        _context: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward_with_mask(&h, mask)?;
        }
        Ok(h)
    }
}

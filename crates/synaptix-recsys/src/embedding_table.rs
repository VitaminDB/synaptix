use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub struct EmbeddingTable {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl EmbeddingTable {
    pub fn new(num_embeddings: usize, embedding_dim: usize, device: Device, dtype: DType) -> Result<Self> {
        let weight = Tensor::zeros(vec![num_embeddings, embedding_dim], dtype, device)?;
        Ok(Self { weight, num_embeddings, embedding_dim })
    }

    /// Lookup строк по индексам: `indices` формы `S` → выход `S + [embedding_dim]`.
    pub fn forward(&self, indices: &Tensor) -> Result<Tensor> {
        self.weight.index_select(0, indices).map_err(crate::error::RecSysError::Core)
    }
}

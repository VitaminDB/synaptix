use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn token_embedding(ids: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if weight.rank() != 2 {
        return Err(SynaptixError::Unsupported("token_embedding: weight must be 2D (V, D)"));
    }
    let d = weight.dims()[1];
    let ids_dims = ids.dims().to_vec();
    let flat_ids = ids.contiguous()?.reshape((ids.numel(),))?;
    let selected = weight.index_select(0, &flat_ids)?;
    let mut out_dims = ids_dims;
    out_dims.push(d);
    selected.reshape(out_dims)
}

#[derive(Debug, Clone)]
pub struct TokenEmbedding {
    weight: Tensor,
}

impl TokenEmbedding {
    pub fn new(weight: Tensor) -> Self { Self { weight } }
    pub fn weight(&self) -> &Tensor { &self.weight }
    pub fn forward(&self, ids: &Tensor) -> Result<Tensor> {
        token_embedding(ids, &self.weight)
    }
}

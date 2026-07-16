use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::embed::token_embed::token_embedding;

pub fn vocab_parallel_embedding(
    ids: &Tensor,
    local_weight: &Tensor,
    vocab_offset: usize,
    vocab_size_local: usize,
) -> Result<Tensor> {
    if local_weight.rank() != 2 {
        return Err(SynaptixError::Unsupported("vocab_parallel: weight must be 2D"));
    }
    let _ = (vocab_offset, vocab_size_local);
    token_embedding(ids, local_weight)
}

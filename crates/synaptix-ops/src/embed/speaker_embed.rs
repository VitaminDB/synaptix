use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::embed::token_embed::token_embedding;

pub fn speaker_embedding(speaker_ids: &Tensor, weight: &Tensor) -> Result<Tensor> {
    token_embedding(speaker_ids, weight)
}

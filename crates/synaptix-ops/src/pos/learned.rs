use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn learned_positional_embedding(positions: &Tensor, table: &Tensor) -> Result<Tensor> {
    table.index_select(0, positions)
}

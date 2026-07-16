//! Sequence parallelism: активации режутся по оси последовательности между
//! рангами. Локальная математика scatter/gather; gather(scatter всех рангов) == x.

use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};
use crate::world::shard_range;

/// Срез `x` вдоль `dim` для ранга `rank` из `world_size` (почти равные части).
pub fn scatter_sequence(x: &Tensor, dim: usize, rank: usize, world_size: usize) -> Result<Tensor> {
    let total = x.dims().get(dim).copied().unwrap_or(0);
    let (off, len) = shard_range(total, rank, world_size);
    x.narrow(dim, off, len).and_then(|t| t.contiguous()).map_err(DistError::Core)
}

/// Собрать шарды последовательности всех рангов обратно по `dim`.
pub fn gather_sequence(shards: &[&Tensor], dim: usize) -> Result<Tensor> {
    if shards.is_empty() {
        return Err(DistError::Other("gather_sequence: empty".into()));
    }
    Tensor::cat(shards, dim).map_err(DistError::Core)
}

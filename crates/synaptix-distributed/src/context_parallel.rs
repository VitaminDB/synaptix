//! Context parallelism: KV-кэш (контекст) режется по оси последовательности
//! между рангами. Локальная математика scatter/gather; gather(scatter) == x.

use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};
use crate::world::shard_range;

/// Срез KV `x` вдоль `dim` для ранга `rank` из `world_size`.
pub fn scatter_kv(x: &Tensor, dim: usize, rank: usize, world_size: usize) -> Result<Tensor> {
    let total = x.dims().get(dim).copied().unwrap_or(0);
    let (off, len) = shard_range(total, rank, world_size);
    x.narrow(dim, off, len).and_then(|t| t.contiguous()).map_err(DistError::Core)
}

/// Собрать KV-шарды всех рангов обратно по `dim`.
pub fn gather_kv(shards: &[&Tensor], dim: usize) -> Result<Tensor> {
    if shards.is_empty() {
        return Err(DistError::Other("gather_kv: empty".into()));
    }
    Tensor::cat(shards, dim).map_err(DistError::Core)
}

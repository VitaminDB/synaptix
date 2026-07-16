use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};
use crate::collectives::all_reduce::{reduce_shards, ReduceOp};
use crate::world::shard_range;

/// Reduce-scatter: свернуть шарды всех рангов (по `op`), затем разрезать
/// результат по `dim` на `shards.len()` частей — каждый ранг получает свою.
/// Возвращает по одному куску на ранг (локальная математика).
pub fn reduce_scatter(shards: &[&Tensor], op: ReduceOp, dim: usize) -> Result<Vec<Tensor>> {
    let world = shards.len();
    if world == 0 {
        return Err(DistError::Other("reduce_scatter: empty".into()));
    }
    let reduced = reduce_shards(shards, op)?;
    let total = reduced.dims().get(dim).copied().unwrap_or(0);
    let mut out = Vec::with_capacity(world);
    for r in 0..world {
        let (off, len) = shard_range(total, r, world);
        let piece = reduced
            .narrow(dim, off, len)
            .and_then(|t| t.contiguous())
            .map_err(DistError::Core)?;
        out.push(piece);
    }
    Ok(out)
}

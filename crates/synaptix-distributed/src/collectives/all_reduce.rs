use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp { Sum, Mean, Max, Min }

/// Per-rank API all-reduce: в реальной системе каждый ранг зовёт это со своим
/// шардом и получает свёрнутый по всем рангам результат (NCCL). В single-process
/// (`world_size==1`) свёртка одного шарда = он сам (для `Mean` — деление на 1).
/// Реальная математика свёртки по нескольким шардам — в [`reduce_shards`].
pub fn all_reduce(tensor: &Tensor, op: ReduceOp) -> Result<Tensor> {
    reduce_shards(&[tensor], op)
}

/// Свёртка нескольких шардов одинаковой формы (локальная математика all-reduce).
pub fn reduce_shards(shards: &[&Tensor], op: ReduceOp) -> Result<Tensor> {
    let first = shards.first().ok_or_else(|| DistError::Other("reduce_shards: empty".into()))?;
    let mut acc = (*first).clone();
    for s in &shards[1..] {
        acc = match op {
            ReduceOp::Sum | ReduceOp::Mean => acc.add(s),
            ReduceOp::Max => acc.maximum(s),
            ReduceOp::Min => acc.minimum(s),
        }
        .map_err(DistError::Core)?;
    }
    if op == ReduceOp::Mean {
        acc = acc.mul_scalar(1.0 / shards.len() as f32).map_err(DistError::Core)?;
    }
    Ok(acc)
}

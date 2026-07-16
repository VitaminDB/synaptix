//! Tensor parallelism (Megatron-style). Реализована локальная математика
//! шардирования: вес режется по рангам, частичные результаты рекомбинируются —
//! итог совпадает с нешардированным `x @ weight`, что и проверяют тесты.
//! Реальный обмен между рангами — через [`crate::collectives`] (NCCL).

use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};
use crate::world::shard_range;

/// Column-parallel linear: вес `[in, out]` режется по столбцам (`out`) на
/// `world_size` шардов, каждый ранг считает `x @ w_shard`, выходы all-gather'ятся
/// по последней оси. Результат == `x @ weight`.
pub fn column_parallel_linear(x: &Tensor, weight: &Tensor, world_size: usize) -> Result<Tensor> {
    let (_in_dim, out_dim) = weight.dims2().map_err(DistError::Core)?;
    let out_axis = x.rank() - 1;
    let mut parts: Vec<Tensor> = Vec::new();
    for r in 0..world_size {
        let (off, len) = shard_range(out_dim, r, world_size);
        if len == 0 {
            continue;
        }
        let w_shard = weight.narrow(1, off, len).and_then(|t| t.contiguous()).map_err(DistError::Core)?;
        parts.push(x.matmul(&w_shard).map_err(DistError::Core)?);
    }
    let refs: Vec<&Tensor> = parts.iter().collect();
    Tensor::cat(&refs, out_axis).map_err(DistError::Core)
}

/// Row-parallel linear: вес `[in, out]` режется по строкам (`in`), вход `x`
/// режется по последней оси на те же шарды; каждый ранг считает частичный
/// `x_shard @ w_shard`, частичные суммы all-reduce'ятся. Результат == `x @ weight`.
pub fn row_parallel_linear(x: &Tensor, weight: &Tensor, world_size: usize) -> Result<Tensor> {
    let (in_dim, _out_dim) = weight.dims2().map_err(DistError::Core)?;
    let in_axis = x.rank() - 1;
    let mut acc: Option<Tensor> = None;
    for r in 0..world_size {
        let (off, len) = shard_range(in_dim, r, world_size);
        if len == 0 {
            continue;
        }
        let w_shard = weight.narrow(0, off, len).and_then(|t| t.contiguous()).map_err(DistError::Core)?;
        let x_shard = x.narrow(in_axis, off, len).and_then(|t| t.contiguous()).map_err(DistError::Core)?;
        let p = x_shard.matmul(&w_shard).map_err(DistError::Core)?;
        acc = Some(match acc {
            Some(a) => a.add(&p).map_err(DistError::Core)?,
            None => p,
        });
    }
    acc.ok_or_else(|| DistError::Other("row_parallel_linear: empty world".into()))
}

//! ZeRO публичный API.
//!
//! Общая математика всех стадий — равномерное разбиение плоского состояния по
//! рангам ([`partition_flat`]) и обратная сборка ([`gather_flat`]). Стадии
//! отличаются тем, ЧТО шардируется (см. zero1/zero2/zero3), но арифметика
//! партиционирования одна.

pub mod offload;
pub mod zero1;
pub mod zero2;
pub mod zero3;

use synaptix_core::tensor::Tensor;
use crate::error::{DistError, Result};
use crate::world::shard_range;

/// Срез плоского состояния (тензор любой формы → flatten → `[numel]`) для
/// `rank` из `world_size`.
pub fn partition_flat(tensor: &Tensor, rank: usize, world_size: usize) -> Result<Tensor> {
    let flat = tensor.flatten_all().map_err(DistError::Core)?;
    let numel = flat.dims().first().copied().unwrap_or(0);
    let (off, len) = shard_range(numel, rank, world_size);
    flat.narrow(0, off, len).and_then(|t| t.contiguous()).map_err(DistError::Core)
}

/// Собрать плоские шарды всех рангов обратно в `[numel]`.
pub fn gather_flat(shards: &[&Tensor]) -> Result<Tensor> {
    if shards.is_empty() {
        return Err(DistError::Other("gather_flat: empty".into()));
    }
    Tensor::cat(shards, 0).map_err(DistError::Core)
}

use synaptix_core::tensor::Tensor;

use crate::error::{DistError, Result};

/// Собрать шарды всех рангов в один тензор конкатенацией по `dim`
/// (локальная математика all-gather).
pub fn all_gather(shards: &[&Tensor], dim: usize) -> Result<Tensor> {
    if shards.is_empty() {
        return Err(DistError::Other("all_gather: empty".into()));
    }
    Tensor::cat(shards, dim).map_err(DistError::Core)
}

/// Per-rank API: реплицировать собственный тензор на `world_size` рангов.
/// Полноценный all-gather с разными шардами — [`all_gather`].
pub fn all_gather_replicate(tensor: &Tensor, world_size: usize) -> Result<Vec<Tensor>> {
    Ok(vec![tensor.clone(); world_size])
}

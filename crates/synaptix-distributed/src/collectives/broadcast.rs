use synaptix_core::tensor::Tensor;

use crate::error::Result;

/// Per-rank API: вернуть тензор источника (в single-process источник локален).
pub fn broadcast(tensor: &Tensor, _src_rank: usize) -> Result<Tensor> {
    Ok(tensor.clone())
}

/// Реплицировать тензор источника на все `world_size` рангов
/// (локальная математика broadcast).
pub fn broadcast_to_all(tensor: &Tensor, world_size: usize) -> Result<Vec<Tensor>> {
    Ok(vec![tensor.clone(); world_size])
}

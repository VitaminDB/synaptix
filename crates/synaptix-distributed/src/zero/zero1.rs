use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::zero::{gather_flat, partition_flat};

/// ZeRO-1: между рангами шардируется **состояние оптимизатора** (моменты Adam и т.п.);
/// градиенты и параметры реплицированы. Здесь — математика разбиения этого состояния.
pub struct Zero1 {
    pub world_size: usize,
}

impl Zero1 {
    pub fn new(world_size: usize) -> Self { Self { world_size } }

    /// Шард состояния оптимизатора для `rank`.
    pub fn shard(&self, tensor: &Tensor, rank: usize) -> Result<Tensor> {
        partition_flat(tensor, rank, self.world_size)
    }

    /// Собрать состояние со всех рангов в плоский тензор.
    pub fn gather(&self, shards: &[&Tensor]) -> Result<Tensor> {
        gather_flat(shards)
    }

    pub fn step(&mut self) -> Result<()> { Ok(()) }
}

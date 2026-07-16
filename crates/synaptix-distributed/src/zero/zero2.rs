use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::zero::{gather_flat, partition_flat};

/// ZeRO-2: шардируются **состояние оптимизатора + градиенты**; параметры
/// реплицированы. Партиционирование то же, что в ZeRO-1, но применяется и к
/// градиентам (reduce-scatter вместо all-reduce).
pub struct Zero2 {
    pub world_size: usize,
}

impl Zero2 {
    pub fn new(world_size: usize) -> Self { Self { world_size } }

    /// Шард градиента/состояния для `rank`.
    pub fn shard(&self, tensor: &Tensor, rank: usize) -> Result<Tensor> {
        partition_flat(tensor, rank, self.world_size)
    }

    pub fn gather(&self, shards: &[&Tensor]) -> Result<Tensor> {
        gather_flat(shards)
    }

    pub fn step(&mut self) -> Result<()> { Ok(()) }
}

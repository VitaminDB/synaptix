use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::zero::{gather_flat, partition_flat};

/// ZeRO-3 (FSDP): шардируются **параметры + градиенты + состояние оптимизатора**.
/// Параметры собираются all-gather'ом только на время forward/backward слоя.
/// Здесь — математика разбиения параметров и их обратной сборки.
pub struct Zero3 {
    pub world_size: usize,
}

impl Zero3 {
    pub fn new(world_size: usize) -> Self { Self { world_size } }

    /// Шард параметра для `rank`.
    pub fn shard(&self, tensor: &Tensor, rank: usize) -> Result<Tensor> {
        partition_flat(tensor, rank, self.world_size)
    }

    /// All-gather параметра: собрать полный плоский тензор из шардов.
    pub fn all_gather_param(&self, shards: &[&Tensor]) -> Result<Tensor> {
        gather_flat(shards)
    }

    pub fn step(&mut self) -> Result<()> { Ok(()) }
}

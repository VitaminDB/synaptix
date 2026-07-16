use crate::error::Result;

pub struct DistributedDataParallel {
    pub rank: usize,
    pub world_size: usize,
}

impl DistributedDataParallel {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self { rank, world_size }
    }

    pub fn sync_gradients(&self) -> Result<()> { Ok(()) }
}

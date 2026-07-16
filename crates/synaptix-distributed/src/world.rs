use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{DistError, Result};

static WORLD_SIZE: AtomicUsize = AtomicUsize::new(1);
static RANK: AtomicUsize = AtomicUsize::new(0);

pub fn init(rank: usize, world_size: usize) -> Result<()> {
    if rank >= world_size {
        return Err(DistError::RankOutOfRange { rank, world_size });
    }
    WORLD_SIZE.store(world_size, Ordering::Relaxed);
    RANK.store(rank, Ordering::Relaxed);
    Ok(())
}

pub fn rank() -> usize { RANK.load(Ordering::Relaxed) }
pub fn world_size() -> usize { WORLD_SIZE.load(Ordering::Relaxed) }
pub fn is_master() -> bool { rank() == 0 }

/// Разбить `n` элементов на `world_size` почти равных непрерывных частей и
/// вернуть `(offset, len)` для `rank`. Остаток `n % world_size` раздаётся по
/// одному ранним рангам. Сумма всех `len` == `n`, части не пересекаются.
pub fn shard_range(n: usize, rank: usize, world_size: usize) -> (usize, usize) {
    if world_size == 0 || rank >= world_size {
        return (0, 0);
    }
    let base = n / world_size;
    let rem = n % world_size;
    let len = base + if rank < rem { 1 } else { 0 };
    let offset = base * rank + rem.min(rank);
    (offset, len)
}

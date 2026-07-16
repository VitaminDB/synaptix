use std::path::Path;

use memmap2::Mmap;

use crate::error::{IoError, Result};
use super::dataloader::Dataset;

pub struct MmapDataset {
    mmap: Mmap,
    record_size: usize,
    count: usize,
}

impl MmapDataset {
    pub fn open(path: impl AsRef<Path>, record_size: usize) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())
            .map_err(|e| IoError::Io(e))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| IoError::Io(e))?;
        let file_len = mmap.len();
        if record_size == 0 {
            return Err(IoError::Data("record_size must be > 0".into()));
        }
        let count = file_len / record_size;
        Ok(Self { mmap, record_size, count })
    }

    pub fn record_bytes(&self, idx: usize) -> Result<&[u8]> {
        if idx >= self.count {
            return Err(IoError::Data(format!("index {idx} out of range (count={})", self.count)));
        }
        let start = idx * self.record_size;
        Ok(&self.mmap[start..start + self.record_size])
    }
}

impl Dataset for MmapDataset {
    type Item = Vec<u8>;

    fn len(&self) -> usize {
        self.count
    }

    fn get(&self, idx: usize) -> Result<Self::Item> {
        Ok(self.record_bytes(idx)?.to_vec())
    }
}

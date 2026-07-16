use std::sync::Arc;

use crate::error::Result;

pub trait Dataset: Send + Sync {
    type Item;
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> Result<Self::Item>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    seed: u64,
}

impl<D: Dataset> DataLoader<D> {
    pub fn new(dataset: Arc<D>, batch_size: usize) -> Self {
        Self { dataset, batch_size, shuffle: false, seed: 0 }
    }

    pub fn with_shuffle(mut self, seed: u64) -> Self {
        self.shuffle = true;
        self.seed = seed;
        self
    }
}

impl<D: Dataset> DataLoader<D>
where
    D::Item: Clone,
{
    pub fn iter(&self) -> DataLoaderIter<D> {
        let n = self.dataset.len();
        let order = if self.shuffle {
            shuffle_indices(n, self.seed)
        } else {
            (0..n).collect()
        };
        DataLoaderIter {
            dataset: Arc::clone(&self.dataset),
            order,
            pos: 0,
            batch_size: self.batch_size,
        }
    }
}

pub struct DataLoaderIter<D: Dataset> {
    dataset: Arc<D>,
    order: Vec<usize>,
    pos: usize,
    batch_size: usize,
}

impl<D: Dataset> Iterator for DataLoaderIter<D>
where
    D::Item: Clone,
{
    type Item = Result<Vec<D::Item>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.order.len() {
            return None;
        }
        let end = (self.pos + self.batch_size).min(self.order.len());
        let mut batch = Vec::with_capacity(end - self.pos);
        for &idx in &self.order[self.pos..end] {
            match self.dataset.get(idx) {
                Ok(item) => batch.push(item),
                Err(e) => {
                    self.pos = end;
                    return Some(Err(e));
                }
            }
        }
        self.pos = end;
        Some(Ok(batch))
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn shuffle_indices(n: usize, seed: u64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    let mut state = seed;
    for i in (1..n).rev() {
        let j = (splitmix64(&mut state) as usize) % (i + 1);
        indices.swap(i, j);
    }
    indices
}

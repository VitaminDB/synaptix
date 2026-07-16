use super::dataloader::Dataset;
use crate::error::Result;

pub struct StreamingDataset<T> {
    items: Vec<T>,
}

impl<T: Clone + Send + Sync> StreamingDataset<T> {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub fn push(&mut self, item: T) {
        self.items.push(item);
    }

    pub fn from_iter(iter: impl Iterator<Item = T>) -> Self {
        Self { items: iter.collect() }
    }
}

impl<T: Clone + Send + Sync> Default for StreamingDataset<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Clone + Send + Sync> Dataset for StreamingDataset<T> {
    type Item = T;

    fn len(&self) -> usize {
        self.items.len()
    }

    fn get(&self, idx: usize) -> Result<Self::Item> {
        Ok(self.items[idx].clone())
    }
}

pub struct ChainedDataset<A: Dataset, B: Dataset<Item = A::Item>> {
    a: A,
    b: B,
}

impl<A: Dataset, B: Dataset<Item = A::Item>> ChainedDataset<A, B> {
    pub fn new(a: A, b: B) -> Self {
        Self { a, b }
    }
}

impl<A: Dataset, B: Dataset<Item = A::Item>> Dataset for ChainedDataset<A, B> {
    type Item = A::Item;

    fn len(&self) -> usize {
        self.a.len() + self.b.len()
    }

    fn get(&self, idx: usize) -> Result<Self::Item> {
        let a_len = self.a.len();
        if idx < a_len {
            self.a.get(idx)
        } else {
            self.b.get(idx - a_len)
        }
    }
}

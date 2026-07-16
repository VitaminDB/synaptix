use synaptix_core::tensor::Tensor;
use crate::error::Result;

pub struct SqliteVecIndex { pub dim: usize }

impl SqliteVecIndex {
    pub fn new(dim: usize) -> Self { Self { dim } }
    pub fn add(&mut self, _id: String, _emb: Tensor) {}
    pub fn search(&self, _query: &Tensor, top_k: usize) -> Result<Vec<(String, f32)>> {
        Ok(Vec::with_capacity(top_k))
    }
}

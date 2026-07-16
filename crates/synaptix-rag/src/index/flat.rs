use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::metric::{cosine, tensor_to_vec, top_k_desc};

/// Плоский (brute-force) индекс: точный поиск косинусной близости по всем
/// эмбеддингам.
pub struct FlatIndex {
    pub embeddings: Vec<Vec<f32>>,
    pub ids: Vec<String>,
    pub dim: usize,
}

impl FlatIndex {
    pub fn new(dim: usize) -> Self { Self { embeddings: Vec::new(), ids: Vec::new(), dim } }

    pub fn add(&mut self, id: String, emb: Tensor) {
        if let Ok(v) = tensor_to_vec(&emb) {
            self.ids.push(id);
            self.embeddings.push(v);
        }
    }

    pub fn search(&self, query: &Tensor, top_k: usize) -> Result<Vec<(String, f32)>> {
        let q = tensor_to_vec(query)?;
        let scored: Vec<(String, f32)> = self
            .ids
            .iter()
            .zip(&self.embeddings)
            .map(|(id, e)| (id.clone(), cosine(&q, e)))
            .collect();
        Ok(top_k_desc(scored, top_k))
    }

    pub fn len(&self) -> usize { self.ids.len() }
    pub fn is_empty(&self) -> bool { self.ids.is_empty() }
}

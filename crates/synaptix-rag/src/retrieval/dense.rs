use synaptix_core::tensor::Tensor;
use crate::error::Result;
use crate::metric::{cosine, tensor_to_vec, top_k_desc};

/// Плотный ретривер: точный поиск по косинусной близости.
pub struct DenseRetriever {
    pub index: Vec<Tensor>,
    pub texts: Vec<String>,
    pub dim: usize,
}

impl DenseRetriever {
    pub fn new(dim: usize) -> Self { Self { index: Vec::new(), texts: Vec::new(), dim } }

    pub fn add(&mut self, text: String, embedding: Tensor) {
        self.texts.push(text);
        self.index.push(embedding);
    }

    pub fn search(&self, query_emb: &Tensor, top_k: usize) -> Result<Vec<(String, f32)>> {
        let q = tensor_to_vec(query_emb)?;
        let mut scored = Vec::with_capacity(self.texts.len());
        for (e, t) in self.index.iter().zip(&self.texts) {
            let ev = tensor_to_vec(e)?;
            scored.push((t.clone(), cosine(&q, &ev)));
        }
        Ok(top_k_desc(scored, top_k))
    }
}

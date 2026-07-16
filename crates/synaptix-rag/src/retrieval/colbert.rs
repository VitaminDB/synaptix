use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use crate::error::{RagError, Result};
use crate::metric::{cosine, top_k_desc};

/// ColBERT late-interaction ретривер: документ хранится матрицей токен-эмбеддингов;
/// релевантность = MaxSim — сумма по токенам запроса максимальной близости к
/// токенам документа.
pub struct ColBert {
    pub dim: usize,
    docs: Vec<(String, Vec<Vec<f32>>)>,
}

impl ColBert {
    pub fn new(dim: usize) -> Self { Self { dim, docs: Vec::new() } }

    /// Добавить документ: `token_embs` формы `[n_tokens, dim]`.
    pub fn add_doc(&mut self, id: String, token_embs: &Tensor) -> Result<()> {
        let rows = token_embs.to_dtype(DType::F32).and_then(|t| t.to_vec2::<f32>()).map_err(RagError::Core)?;
        self.docs.push((id, rows));
        Ok(())
    }

    /// Поиск по матрице токен-эмбеддингов запроса `[n_q, dim]`.
    pub fn search(&self, query_embs: &Tensor, top_k: usize) -> Result<Vec<(String, f32)>> {
        let q = query_embs.to_dtype(DType::F32).and_then(|t| t.to_vec2::<f32>()).map_err(RagError::Core)?;
        let scored: Vec<(String, f32)> = self
            .docs
            .iter()
            .map(|(id, doc)| (id.clone(), max_sim(&q, doc)))
            .collect();
        Ok(top_k_desc(scored, top_k))
    }

    pub fn len(&self) -> usize { self.docs.len() }
    pub fn is_empty(&self) -> bool { self.docs.is_empty() }
}

/// MaxSim: Σ_q max_d cos(q, d). Пустой документ даёт 0.
fn max_sim(query: &[Vec<f32>], doc: &[Vec<f32>]) -> f32 {
    if doc.is_empty() {
        return 0.0;
    }
    query
        .iter()
        .map(|qt| doc.iter().map(|dt| cosine(qt, dt)).fold(f32::NEG_INFINITY, f32::max))
        .sum()
}

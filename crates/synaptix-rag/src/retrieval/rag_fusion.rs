use std::collections::HashMap;
use crate::error::Result;
use crate::metric::top_k_desc;

/// RAG-Fusion: несколько перефразировок запроса дают несколько ранжированных
/// списков, которые объединяются Reciprocal Rank Fusion (RRF).
///
/// Генерация перефразировок требует LLM (вне этого крейта) — здесь реализовано
/// ядро слияния [`RagFusion::reciprocal_rank_fusion`]; `num_queries` хранит
/// ожидаемое число запросов.
pub struct RagFusion {
    pub num_queries: usize,
    /// Константа RRF (обычно 60): score(d) = Σ_l 1/(k + rank_l(d)).
    pub k_const: f32,
}

impl Default for RagFusion {
    fn default() -> Self { Self { num_queries: 4, k_const: 60.0 } }
}

impl RagFusion {
    pub fn new(num_queries: usize) -> Self {
        Self { num_queries, k_const: 60.0 }
    }

    /// Reciprocal Rank Fusion по нескольким ранжированным спискам. Внутри каждого
    /// списка порядок задаёт ранг (1 — топ); score(id) = Σ 1/(k_const + rank).
    pub fn reciprocal_rank_fusion(
        &self,
        result_lists: &[Vec<(String, f32)>],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let mut fused: HashMap<String, f32> = HashMap::new();
        for list in result_lists {
            for (rank, (id, _score)) in list.iter().enumerate() {
                let contrib = 1.0 / (self.k_const + (rank + 1) as f32);
                *fused.entry(id.clone()).or_insert(0.0) += contrib;
            }
        }
        Ok(top_k_desc(fused.into_iter().collect(), top_k))
    }
}

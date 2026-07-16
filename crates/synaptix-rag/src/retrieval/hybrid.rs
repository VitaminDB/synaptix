use std::collections::HashMap;
use crate::error::Result;
use crate::metric::top_k_desc;

/// Гибридный ретривер: взвешенная фьюжн плотных (dense) и разреженных (sparse,
/// напр. BM25) скоров. Каждый список min-max нормализуется в [0,1], затем
/// `dense_weight·dense + sparse_weight·sparse` по объединению id.
pub struct HybridRetriever {
    pub dense_weight: f32,
    pub sparse_weight: f32,
}

impl Default for HybridRetriever {
    fn default() -> Self { Self { dense_weight: 0.5, sparse_weight: 0.5 } }
}

impl HybridRetriever {
    pub fn new(dense_weight: f32, sparse_weight: f32) -> Self {
        Self { dense_weight, sparse_weight }
    }

    /// Слить два ранжированных списка `(id, score)` в один.
    pub fn fuse(
        &self,
        dense: &[(String, f32)],
        sparse: &[(String, f32)],
        top_k: usize,
    ) -> Result<Vec<(String, f32)>> {
        let dn = min_max_norm(dense);
        let sp = min_max_norm(sparse);
        let mut fused: HashMap<String, f32> = HashMap::new();
        for (id, s) in &dn {
            *fused.entry(id.clone()).or_insert(0.0) += self.dense_weight * s;
        }
        for (id, s) in &sp {
            *fused.entry(id.clone()).or_insert(0.0) += self.sparse_weight * s;
        }
        Ok(top_k_desc(fused.into_iter().collect(), top_k))
    }
}

/// Нормировать скоры списка в [0,1]. Если все равны — все 1.0.
fn min_max_norm(list: &[(String, f32)]) -> Vec<(String, f32)> {
    if list.is_empty() {
        return Vec::new();
    }
    let min = list.iter().map(|x| x.1).fold(f32::INFINITY, f32::min);
    let max = list.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    list.iter()
        .map(|(id, s)| (id.clone(), if range > 0.0 { (s - min) / range } else { 1.0 }))
        .collect()
}

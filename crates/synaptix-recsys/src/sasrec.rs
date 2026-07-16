//! SASRec (Self-Attentive Sequential Recommendation): причинное (causal)
//! self-attention над последовательностью эмбеддингов товаров; скор следующего
//! товара = dot последнего скрытого состояния со всеми эмбеддингами товаров.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;

use crate::embedding_table::EmbeddingTable;
use crate::error::{RecSysError, Result};
use crate::util::softmax_vec;

pub struct SasrecConfig {
    pub emb_dim: usize,
    pub num_embeddings: usize,
}

impl Default for SasrecConfig {
    fn default() -> Self { Self { emb_dim: 64, num_embeddings: 1024 } }
}

pub struct Sasrec {
    pub emb: EmbeddingTable,
}

impl Sasrec {
    pub fn new(config: SasrecConfig, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self { emb: EmbeddingTable::new(config.num_embeddings, config.emb_dim, device, dtype)? })
    }

    pub fn from_emb(emb: EmbeddingTable) -> Self {
        Self { emb }
    }

    /// Причинное self-attention над `[L, d]` (позиция `i` видит только `j ≤ i`).
    /// Q=K=V=входу; веса = softmax(QKᵀ/√d) с causal-маской. Возвращает `[L, d]`.
    pub fn self_attention(&self, embs: &Tensor) -> Result<Tensor> {
        let e = embs.to_vec2::<f32>().map_err(RecSysError::Core)?;
        let l = e.len();
        if l == 0 {
            return Ok(embs.clone());
        }
        let d = e[0].len();
        let scale = 1.0 / (d as f32).sqrt();
        let mut out = vec![vec![0.0f32; d]; l];
        for i in 0..l {
            // Скоры только по j ≤ i (causal).
            let scores: Vec<f32> = (0..=i).map(|j| dot(&e[i], &e[j]) * scale).collect();
            let w = softmax_vec(&scores);
            for (j, &wj) in w.iter().enumerate() {
                for k in 0..d {
                    out[i][k] += wj * e[j][k];
                }
            }
        }
        let flat: Vec<f32> = out.into_iter().flatten().collect();
        Tensor::from_vec::<_, f32>(flat, vec![l, d], embs.device()).map_err(RecSysError::Core)
    }

    /// `item_ids [L]` → логиты по всем товарам `[num_embeddings]` (скор следующего
    /// товара по последнему скрытому состоянию).
    pub fn forward(&self, item_ids: &Tensor) -> Result<Tensor> {
        let embs = self.emb.forward(item_ids)?; // [L, d]
        let att = self.self_attention(&embs)?; // [L, d]
        let l = att.dims()[0];
        let d = att.dims()[1];
        let last = att.narrow(0, l - 1, 1).and_then(|t| t.contiguous()).map_err(RecSysError::Core)?; // [1, d]
        let last_col = last.transpose(0, 1).and_then(|t| t.contiguous()).map_err(RecSysError::Core)?; // [d, 1]
        let logits = self.emb.weight.matmul(&last_col).map_err(RecSysError::Core)?; // [num_emb, 1]
        let _ = d;
        logits.squeeze(1).map_err(RecSysError::Core)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

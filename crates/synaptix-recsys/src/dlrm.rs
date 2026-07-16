//! DLRM (Deep Learning Recommendation Model): bottom-MLP над плотными фичами →
//! плотный эмбеддинг; разреженные фичи → lookup; попарные dot-произведения всех
//! эмбеддингов (interaction) конкатенируются с плотным эмбеддингом → top-MLP → логит.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::init::InitMethod;
use synaptix_nn::linear::Linear;

use crate::embedding_table::EmbeddingTable;
use crate::error::{RecSysError, Result};
use crate::util::apply_mlp;

pub struct DlrmConfig {
    pub dense_in: usize,
    pub emb_dim: usize,
    pub n_sparse: usize,
    pub num_embeddings: usize,
    pub hidden_dim: usize,
}

impl Default for DlrmConfig {
    fn default() -> Self {
        Self { dense_in: 13, emb_dim: 16, n_sparse: 26, num_embeddings: 1024, hidden_dim: 64 }
    }
}

pub struct Dlrm {
    pub bottom: Vec<Linear>,
    pub emb: EmbeddingTable,
    pub top: Vec<Linear>,
}

impl Dlrm {
    pub fn new(config: DlrmConfig, device: Device, dtype: DType) -> Result<Self> {
        let mk = |inp, out, seed| {
            Linear::from_init(inp, out, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, seed)
                .map_err(RecSysError::Core)
        };
        let fields = 1 + config.n_sparse; // dense_emb + sparse
        let n_pairs = fields * (fields - 1) / 2;
        let top_in = config.emb_dim + n_pairs;
        Ok(Self {
            bottom: vec![mk(config.dense_in, config.hidden_dim, 0)?, mk(config.hidden_dim, config.emb_dim, 1)?],
            emb: EmbeddingTable::new(config.num_embeddings, config.emb_dim, device, dtype)?,
            top: vec![mk(top_in, config.hidden_dim, 2)?, mk(config.hidden_dim, 1, 3)?],
        })
    }

    /// Конструктор с явными слоями (тесты / загрузка весов).
    pub fn from_layers(bottom: Vec<Linear>, emb: EmbeddingTable, top: Vec<Linear>) -> Self {
        Self { bottom, emb, top }
    }

    /// `dense [B, dense_in]`, `sparse` — по одному тензору индексов `[B]` на разреженную фичу.
    /// Возвращает логиты `[B, 1]`.
    pub fn forward(&self, dense: &Tensor, sparse: &[Tensor]) -> Result<Tensor> {
        let dense_emb = apply_mlp(&self.bottom, dense)?; // [B, emb_dim]
        let mut vecs: Vec<Tensor> = Vec::with_capacity(1 + sparse.len());
        vecs.push(dense_emb.clone());
        for idx in sparse {
            vecs.push(self.emb.forward(idx)?); // [B, emb_dim]
        }
        // Попарные dot-произведения (верхний треугольник).
        let mut cols: Vec<Tensor> = vec![dense_emb];
        for i in 0..vecs.len() {
            for j in (i + 1)..vecs.len() {
                let dot = vecs[i]
                    .mul(&vecs[j])
                    .and_then(|p| p.sum([1])) // [B]
                    .and_then(|s| s.unsqueeze(1)) // [B,1]
                    .map_err(RecSysError::Core)?;
                cols.push(dot);
            }
        }
        let refs: Vec<&Tensor> = cols.iter().collect();
        let top_in = Tensor::cat(&refs, 1).map_err(RecSysError::Core)?; // [B, emb_dim + n_pairs]
        apply_mlp(&self.top, &top_in)
    }
}

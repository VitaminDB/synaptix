//! DIN (Deep Interest Network): target-attention пулинг по истории поведения
//! пользователя относительно целевого товара, затем MLP над `[pooled, target]`.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::init::InitMethod;
use synaptix_nn::linear::Linear;

use crate::embedding_table::EmbeddingTable;
use crate::error::{RecSysError, Result};
use crate::util::{apply_mlp, softmax_vec};

pub struct DinConfig {
    pub emb_dim: usize,
    pub hidden_dim: usize,
    pub num_embeddings: usize,
}

impl Default for DinConfig {
    fn default() -> Self { Self { emb_dim: 16, hidden_dim: 64, num_embeddings: 1024 } }
}

pub struct Din {
    pub emb: EmbeddingTable,
    /// MLP над конкатенацией `[pooled_interest, target]`: `2*emb_dim -> hidden -> 1`.
    pub mlp: Vec<Linear>,
}

impl Din {
    pub fn new(config: DinConfig, device: Device, dtype: DType) -> Result<Self> {
        let mk = |inp, out, seed| {
            Linear::from_init(inp, out, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, seed)
                .map_err(RecSysError::Core)
        };
        Ok(Self {
            emb: EmbeddingTable::new(config.num_embeddings, config.emb_dim, device, dtype)?,
            mlp: vec![mk(2 * config.emb_dim, config.hidden_dim, 0)?, mk(config.hidden_dim, 1, 1)?],
        })
    }

    pub fn from_layers(emb: EmbeddingTable, mlp: Vec<Linear>) -> Self {
        Self { emb, mlp }
    }

    /// Взвешенный пулинг истории: веса = softmax(dot(target, behavior)/√d).
    /// `target [1, d]`, `behaviors [L, d]` → `pooled [1, d]`.
    pub fn attention_pool(&self, target: &Tensor, behaviors: &Tensor) -> Result<Tensor> {
        let t = target.to_vec2::<f32>().map_err(RecSysError::Core)?;
        let b = behaviors.to_vec2::<f32>().map_err(RecSysError::Core)?;
        let tvec = &t[0];
        let d = tvec.len();
        let scale = 1.0 / (d as f32).sqrt();
        let scores: Vec<f32> = b.iter().map(|bi| dot(tvec, bi) * scale).collect();
        let w = softmax_vec(&scores);
        let mut pooled = vec![0.0f32; d];
        for (wi, bi) in w.iter().zip(&b) {
            for k in 0..d {
                pooled[k] += wi * bi[k];
            }
        }
        Tensor::from_vec::<_, f32>(pooled, vec![1, d], target.device()).map_err(RecSysError::Core)
    }

    /// Скор пары (target, история) → `[1, 1]`.
    pub fn forward(&self, target: &Tensor, behaviors: &Tensor) -> Result<Tensor> {
        let pooled = self.attention_pool(target, behaviors)?;
        let x = Tensor::cat(&[&pooled, target], 1).map_err(RecSysError::Core)?;
        apply_mlp(&self.mlp, &x)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

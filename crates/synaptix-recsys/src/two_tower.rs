use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::init::InitMethod;
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

use crate::error::{RecSysError, Result};

pub struct TwoTowerConfig {
    pub query_dim: usize,
    pub item_dim: usize,
    pub hidden_dim: usize,
}

impl Default for TwoTowerConfig {
    fn default() -> Self { Self { query_dim: 256, item_dim: 256, hidden_dim: 128 } }
}

pub struct TwoTower {
    pub query_proj: Linear,
    pub item_proj: Linear,
}

impl TwoTower {
    pub fn new(config: TwoTowerConfig, device: Device, dtype: DType) -> Result<Self> {
        Ok(Self {
            query_proj: Linear::from_init(config.query_dim, config.hidden_dim, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 0).map_err(RecSysError::Core)?,
            item_proj: Linear::from_init(config.item_dim, config.hidden_dim, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, 1).map_err(RecSysError::Core)?,
        })
    }

    /// Конструктор с явными башнями (для тестов / загрузки весов).
    pub fn from_towers(query_proj: Linear, item_proj: Linear) -> Self {
        Self { query_proj, item_proj }
    }

    pub fn embed_query(&self, query: &Tensor) -> Result<Tensor> {
        self.query_proj.forward(query).map_err(RecSysError::Core)
    }

    pub fn embed_item(&self, item: &Tensor) -> Result<Tensor> {
        self.item_proj.forward(item).map_err(RecSysError::Core)
    }

    /// Релевантность пары = скалярное произведение эмбеддингов башен.
    /// `query`/`item` — одиночные образцы (`[query_dim]` или `[1, query_dim]`).
    pub fn score(&self, query: &Tensor, item: &Tensor) -> Result<f32> {
        let q = self.embed_query(query)?;
        let i = self.embed_item(item)?;
        let prod = q.mul(&i).map_err(RecSysError::Core)?;
        let flat = prod.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(RecSysError::Core)?;
        Ok(flat.iter().sum())
    }

    /// Батч скоров: `queries [B, qd]`, `items [B, id]` → `[B]` построчных дотов.
    pub fn scores(&self, queries: &Tensor, items: &Tensor) -> Result<Tensor> {
        let q = self.embed_query(queries)?;
        let i = self.embed_item(items)?;
        q.mul(&i).and_then(|p| p.sum([1])).map_err(RecSysError::Core)
    }
}

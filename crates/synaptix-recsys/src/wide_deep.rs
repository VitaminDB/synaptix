//! Wide & Deep: линейная «wide» часть (запоминание) + «deep» MLP (обобщение),
//! логиты суммируются.

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_nn::init::InitMethod;
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

use crate::error::{RecSysError, Result};
use crate::util::apply_mlp;

pub struct WideDeepConfig {
    pub wide_dim: usize,
    pub deep_in: usize,
    pub hidden_dim: usize,
}

impl Default for WideDeepConfig {
    fn default() -> Self { Self { wide_dim: 64, deep_in: 128, hidden_dim: 256 } }
}

pub struct WideDeep {
    /// Линейная wide-часть: `wide_dim -> 1`.
    pub wide: Linear,
    /// Deep MLP: `deep_in -> hidden -> 1`.
    pub deep: Vec<Linear>,
}

impl WideDeep {
    pub fn new(config: WideDeepConfig, device: Device, dtype: DType) -> Result<Self> {
        let mk = |inp, out, seed| {
            Linear::from_init(inp, out, false, InitMethod::Zeros, InitMethod::Zeros, device, dtype, seed)
                .map_err(RecSysError::Core)
        };
        Ok(Self {
            wide: mk(config.wide_dim, 1, 0)?,
            deep: vec![mk(config.deep_in, config.hidden_dim, 1)?, mk(config.hidden_dim, 1, 2)?],
        })
    }

    /// Конструктор с явными слоями (тесты / загрузка весов).
    pub fn from_layers(wide: Linear, deep: Vec<Linear>) -> Self {
        Self { wide, deep }
    }

    /// `wide_x [B, wide_dim]`, `deep_x [B, deep_in]` → логиты `[B, 1]`.
    pub fn forward(&self, wide_x: &Tensor, deep_x: &Tensor) -> Result<Tensor> {
        let wide_logit = self.wide.forward(wide_x).map_err(RecSysError::Core)?;
        let deep_logit = apply_mlp(&self.deep, deep_x)?;
        wide_logit.add(&deep_logit).map_err(RecSysError::Core)
    }
}

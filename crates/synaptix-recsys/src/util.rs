//! Общие хелперы для рекомендательных моделей.

use synaptix_core::tensor::Tensor;
use synaptix_nn::linear::Linear;
use synaptix_nn::module::Module;

use crate::error::{RecSysError, Result};

/// ReLU через `max(x, 0)` (zeros_like + maximum — без зависимости от unary-op).
pub fn relu(x: &Tensor) -> Result<Tensor> {
    let z = x.zeros_like().map_err(RecSysError::Core)?;
    x.maximum(&z).map_err(RecSysError::Core)
}

/// Численно устойчивый softmax вектора.
pub fn softmax_vec(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() {
        return Vec::new();
    }
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scores.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum <= 0.0 {
        return vec![1.0 / scores.len() as f32; scores.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

/// Прогнать MLP (стек Linear) с ReLU между слоями. Последний слой без активации.
pub fn apply_mlp(layers: &[Linear], x: &Tensor) -> Result<Tensor> {
    let mut h = x.clone();
    let n = layers.len();
    for (i, l) in layers.iter().enumerate() {
        h = l.forward(&h).map_err(RecSysError::Core)?;
        if i + 1 < n {
            h = relu(&h)?;
        }
    }
    Ok(h)
}

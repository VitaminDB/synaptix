//! Векторные метрики для индексов/ретриверов (CPU, f32).

use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use crate::error::{RagError, Result};

/// Развернуть тензор-эмбеддинг любой формы (`[d]`, `[1,d]`, …) в плоский `Vec<f32>`.
pub fn tensor_to_vec(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(RagError::Core)
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Косинусная близость. Нулевой вектор даёт 0.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let denom = norm(a) * norm(b);
    if denom <= 0.0 {
        0.0
    } else {
        dot(a, b) / denom
    }
}

/// Квадрат евклидова расстояния.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Отсортировать `(id, score)` по убыванию score (детерминированно: при равенстве
/// сохраняется исходный порядок через стабильную сортировку) и взять top_k.
pub fn top_k_desc(mut scored: Vec<(String, f32)>, top_k: usize) -> Vec<(String, f32)> {
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
}

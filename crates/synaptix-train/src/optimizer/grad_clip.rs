use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use crate::error::Result;

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    Ok(t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()?)
}

/// Глобальный clip по норме (как `torch.nn.utils.clip_grad_norm_`): считает
/// `total_norm = sqrt(Σ ‖g‖²)` по всем градиентам; если `total_norm > max_norm`,
/// масштабирует все градиенты на `max_norm / (total_norm + 1e-6)`.
/// Возвращает `total_norm` ДО клиппинга.
pub fn clip_grad_norm(grads: &mut [Tensor], max_norm: f64) -> Result<f64> {
    let mut sum_sq = 0.0f64;
    for g in grads.iter() {
        for v in f32v(g)? {
            sum_sq += (v as f64) * (v as f64);
        }
    }
    let total_norm = sum_sq.sqrt();
    if total_norm > max_norm {
        let scale = (max_norm / (total_norm + 1e-6)) as f32;
        for g in grads.iter_mut() {
            *g = g.mul_scalar(scale)?;
        }
    }
    Ok(total_norm)
}

/// Поэлементный clip градиентов в `[-max_val, max_val]`
/// (как `torch.nn.utils.clip_grad_value_`).
pub fn clip_grad_value(grads: &mut [Tensor], max_val: f64) -> Result<()> {
    let mv = max_val as f32;
    for g in grads.iter_mut() {
        let dtype_in = g.dtype();
        let dev = g.device();
        let dims = g.dims().to_vec();
        let clamped: Vec<f32> = f32v(g)?.into_iter().map(|v| v.clamp(-mv, mv)).collect();
        *g = Tensor::from_vec::<_, f32>(clamped, dims, dev)?.to_dtype(dtype_in)?;
    }
    Ok(())
}

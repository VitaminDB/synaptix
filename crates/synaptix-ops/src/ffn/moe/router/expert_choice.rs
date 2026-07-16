use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Expert-choice routing: каждый эксперт выбирает `capacity` токенов с
/// наибольшими логитами. `logits:[N,E]`. Возвращает маску назначений `[N,E]`
/// (1.0 — токен n выбран экспертом e, иначе 0.0).
pub fn expert_choice_router(logits: &Tensor, capacity: usize) -> Result<Tensor> {
    if logits.rank() != 2 {
        return Err(SynaptixError::Unsupported("expert_choice_router: logits must be [N,E]"));
    }
    let (n, e) = (logits.dims()[0], logits.dims()[1]);
    let cap = capacity.min(n);
    let l = f32v(logits)?;
    let mut mask = vec![0.0f32; n * e];
    for j in 0..e {
        // токены, отсортированные по логиту эксперта j (по убыванию)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_unstable_by(|&a, &b| {
            l[b * e + j].partial_cmp(&l[a * e + j]).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &tok in &order[..cap] {
            mask[tok * e + j] = 1.0;
        }
    }
    Tensor::from_vec::<_, f32>(mask, vec![n, e], logits.device())?.to_dtype(logits.dtype())
}

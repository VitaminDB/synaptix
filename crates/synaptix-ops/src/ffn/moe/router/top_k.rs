use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Top-K routing. Для каждого токена выбираются `k` экспертов с наибольшими
/// логитами; веса = softmax по выбранным `k` логитам (перенормировка только по
/// top-k). `logits:[N,E]`. Возвращает `(indices, weights)`, обе формы `[N,k]`
/// (индексы экспертов как f32).
pub fn top_k_router(logits: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    if logits.rank() != 2 {
        return Err(SynaptixError::Unsupported("top_k_router: logits must be [N,E]"));
    }
    let (n, e) = (logits.dims()[0], logits.dims()[1]);
    if k == 0 || k > e {
        return Err(SynaptixError::Unsupported("top_k_router: requires 1 <= k <= E"));
    }
    let l = f32v(logits)?;
    let mut idx_out = vec![0.0f32; n * k];
    let mut w_out = vec![0.0f32; n * k];
    for i in 0..n {
        let row = &l[i * e..i * e + e];
        // индексы по убыванию логита
        let mut order: Vec<usize> = (0..e).collect();
        order.sort_unstable_by(|&a, &b| {
            row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let top = &order[..k];
        // softmax по выбранным k логитам
        let max = top.iter().map(|&j| row[j]).fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = top.iter().map(|&j| (row[j] - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (pos, (&j, &ex)) in top.iter().zip(exps.iter()).enumerate() {
            idx_out[i * k + pos] = j as f32;
            w_out[i * k + pos] = ex / sum;
        }
    }
    let dev = logits.device();
    let indices = Tensor::from_vec::<_, f32>(idx_out, vec![n, k], dev)?;
    let weights = Tensor::from_vec::<_, f32>(w_out, vec![n, k], dev)?;
    Ok((indices, weights))
}

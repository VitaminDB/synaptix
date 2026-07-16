use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Scatter токенов по перестановке `indices`: `out[i] = x[indices[i]]`
/// (сбор строк по индексам — раскладка токенов под экспертов).
/// `x:[N,D]`, `indices:[N]` (целые, любой dtype).
pub fn scatter_tokens(x: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if x.rank() != 2 {
        return Err(SynaptixError::Unsupported("scatter_tokens: x must be [N,D]"));
    }
    let (n, d) = (x.dims()[0], x.dims()[1]);
    let idx = f32v(indices)?;
    if idx.len() != n {
        return Err(SynaptixError::Unsupported("scatter_tokens: indices.len() != N"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let src = idx[i].round() as usize;
        if src >= n {
            return Err(SynaptixError::Unsupported("scatter_tokens: index out of range"));
        }
        out[i * d..i * d + d].copy_from_slice(&xf[src * d..src * d + d]);
    }
    Tensor::from_vec::<_, f32>(out, vec![n, d], x.device())?.to_dtype(dtype_in)
}

/// Обратная операция к [`scatter_tokens`]: `out[indices[i]] = x[i]`
/// (возврат выходов экспертов на исходные позиции токенов).
pub fn gather_tokens(x: &Tensor, indices: &Tensor) -> Result<Tensor> {
    if x.rank() != 2 {
        return Err(SynaptixError::Unsupported("gather_tokens: x must be [N,D]"));
    }
    let (n, d) = (x.dims()[0], x.dims()[1]);
    let idx = f32v(indices)?;
    if idx.len() != n {
        return Err(SynaptixError::Unsupported("gather_tokens: indices.len() != N"));
    }
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let mut out = vec![0.0f32; n * d];
    for i in 0..n {
        let dst = idx[i].round() as usize;
        if dst >= n {
            return Err(SynaptixError::Unsupported("gather_tokens: index out of range"));
        }
        out[dst * d..dst * d + d].copy_from_slice(&xf[i * d..i * d + d]);
    }
    Tensor::from_vec::<_, f32>(out, vec![n, d], x.device())?.to_dtype(dtype_in)
}

use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Titans long-term memory: гейтированное обновление памяти с data-dependent
/// забыванием. Вход `x` задаёт долю удержания `r = sigmoid(x)`, сигнал
/// `surprise` подмешивается в память:
///   `mem_new = r ⊙ mem + (1 − r) ⊙ surprise`.
/// `x, mem, surprise:[B,D]`. Возвращает обновлённую память `[B,D]`.
pub fn titans_memory_step(x: &Tensor, mem: &Tensor, surprise: &Tensor) -> Result<Tensor> {
    if x.rank() != 2 {
        return Err(SynaptixError::Unsupported("titans: x must be rank-2 [B,D]"));
    }
    if mem.dims() != x.dims() || surprise.dims() != x.dims() {
        return Err(SynaptixError::shape_mismatch(x.dims(), mem.dims()));
    }
    let (bsz, d) = (x.dims()[0], x.dims()[1]);
    let dtype_in = x.dtype();
    let xf = f32v(x)?;
    let mf = f32v(mem)?;
    let sf = f32v(surprise)?;

    let mut out = vec![0.0f32; bsz * d];
    for idx in 0..bsz * d {
        let r = 1.0 / (1.0 + (-xf[idx]).exp()); // sigmoid(x)
        out[idx] = r * mf[idx] + (1.0 - r) * sf[idx];
    }
    Tensor::from_vec::<_, f32>(out, vec![bsz, d], x.device())?.to_dtype(dtype_in)
}

use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Soft routing (без жёсткого гейта): softmax по экспертам — веса для взвешенной
/// суммы всех экспертов. `logits:[N,E]` → `[N,E]`.
pub fn soft_router(logits: &Tensor) -> Result<Tensor> {
    if logits.rank() != 2 {
        return Err(SynaptixError::Unsupported("soft_router: logits must be [N,E]"));
    }
    let (n, e) = (logits.dims()[0], logits.dims()[1]);
    let l = f32v(logits)?;
    let mut out = vec![0.0f32; n * e];
    for i in 0..n {
        let row = &l[i * e..i * e + e];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for j in 0..e {
            out[i * e + j] = exps[j] / sum;
        }
    }
    Tensor::from_vec::<_, f32>(out, vec![n, e], logits.device())?.to_dtype(logits.dtype())
}

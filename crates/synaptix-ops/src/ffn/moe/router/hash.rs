use synaptix_core::{
    dtype::DType,
    error::{Result, SynaptixError},
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Hash-based детерминированное routing: `expert = token_id mod num_experts`.
/// `token_ids:[N]` (целые), возвращает `[N]` id экспертов (f32).
pub fn hash_router(token_ids: &Tensor, num_experts: usize) -> Result<Tensor> {
    if num_experts == 0 {
        return Err(SynaptixError::Unsupported("hash_router: num_experts must be > 0"));
    }
    let ids = f32v(token_ids)?;
    let out: Vec<f32> = ids
        .iter()
        .map(|&v| ((v.round() as i64).rem_euclid(num_experts as i64)) as f32)
        .collect();
    let n = out.len();
    Tensor::from_vec::<_, f32>(out, vec![n], token_ids.device())
}

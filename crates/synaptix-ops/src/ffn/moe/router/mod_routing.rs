use synaptix_core::{
    dtype::DType,
    error::Result,
    tensor::Tensor,
};

fn f32v(t: &Tensor) -> Result<Vec<f32>> {
    t.to_dtype(DType::F32)?.contiguous()?.flatten_all()?.to_vec1::<f32>()
}

/// Mixture-of-Depths routing: токен обрабатывается слоем, если его router-скор
/// `≥ depth_threshold`, иначе пропускается (residual). `logits:[N]` (скор на токен).
/// Возвращает `(mask, any_processed)`: `mask:[N]` (1.0 — обработать, 0.0 — пропустить)
/// и флаг, обрабатывается ли хотя бы один токен.
pub fn mod_router(logits: &Tensor, depth_threshold: f32) -> Result<(Tensor, bool)> {
    let l = f32v(logits)?;
    let mut any = false;
    let out: Vec<f32> = l
        .iter()
        .map(|&v| {
            if v >= depth_threshold {
                any = true;
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let n = out.len();
    let mask = Tensor::from_vec::<_, f32>(out, vec![n], logits.device())?
        .to_dtype(logits.dtype())?;
    Ok((mask, any))
}

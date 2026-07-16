use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax_dim;

pub fn scaled_dot_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.rank() < 2 || k.rank() < 2 || v.rank() < 2 {
        return Err(SynaptixError::Unsupported(
            "scaled_dot_attention: rank must be >= 2 (..., S, D)",
        ));
    }
    if q.dtype() != k.dtype() || q.dtype() != v.dtype() {
        return Err(SynaptixError::dtype_mismatch(q.dtype(), k.dtype()));
    }
    let dtype_in = q.dtype();
    let q_f32 = q.to_dtype(DType::F32)?;
    let k_f32 = k.to_dtype(DType::F32)?;
    let v_f32 = v.to_dtype(DType::F32)?;
    let k_rank = k_f32.rank();
    let k_t = k_f32.transpose(k_rank - 2, k_rank - 1)?.contiguous()?;
    let scores = q_f32.matmul(&k_t)?.mul_scalar(scale)?;
    let masked = match mask {
        Some(m) => scores.broadcast_add(&m.to_dtype(DType::F32)?)?,
        None => scores,
    };
    let last = masked.rank() - 1;
    let probs = softmax_dim(&masked, last)?;
    let out = probs.matmul(&v_f32)?;
    out.to_dtype(dtype_in)
}

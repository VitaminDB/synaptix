use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

pub fn cross_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.rank() < 3 || k.rank() < 3 || v.rank() < 3 {
        return Err(SynaptixError::Unsupported(
            "cross_attention: rank must be >= 3 (..., S, D)",
        ));
    }
    let q_heads = q.dims()[q.rank() - 3];
    let k_heads = k.dims()[k.rank() - 3];
    if q_heads == k_heads {
        scaled_dot_attention(q, k, v, scale, mask)
    } else if q_heads % k_heads == 0 {
        let repeats = q_heads / k_heads;
        let k_rep = k.repeat_interleave(k.rank() - 3, repeats)?;
        let v_rep = v.repeat_interleave(v.rank() - 3, repeats)?;
        scaled_dot_attention(q, &k_rep, &v_rep, scale, mask)
    } else {
        Err(SynaptixError::Other(format!(
            "cross_attention: q_heads {q_heads} not divisible by k_heads {k_heads}"
        )))
    }
}

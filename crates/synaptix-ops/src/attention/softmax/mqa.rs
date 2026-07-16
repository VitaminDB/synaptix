use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

pub fn mqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("mqa: requires rank-4 [B,H,S,D]"));
    }
    if k.dims()[1] != 1 || v.dims()[1] != 1 {
        return Err(SynaptixError::Unsupported("mqa: K/V must have 1 head"));
    }
    let h = q.dims()[1];
    let k_expanded = k.expand(vec![q.dims()[0], h, k.dims()[2], k.dims()[3]])?.contiguous()?;
    let v_expanded = v.expand(vec![q.dims()[0], h, v.dims()[2], v.dims()[3]])?.contiguous()?;
    scaled_dot_attention(q, &k_expanded, &v_expanded, scale, mask)
}

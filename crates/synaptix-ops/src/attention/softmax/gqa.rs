use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;

pub fn gqa_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    if q.rank() != 4 || k.rank() != 4 || v.rank() != 4 {
        return Err(SynaptixError::Unsupported("gqa: requires rank-4 [B,H,S,D]"));
    }
    let h_q = q.dims()[1];
    let h_kv = k.dims()[1];
    if v.dims()[1] != h_kv {
        return Err(SynaptixError::shape_mismatch(k.dims(), v.dims()));
    }
    if h_kv == 0 || h_q % h_kv != 0 {
        return Err(SynaptixError::Other(format!(
            "gqa: h_q={h_q} not divisible by h_kv={h_kv}"
        )));
    }
    if h_q == h_kv {
        return scaled_dot_attention(q, k, v, scale, mask);
    }
    let repeats = h_q / h_kv;
    let k_rep = k.repeat_interleave(1, repeats)?;
    let v_rep = v.repeat_interleave(1, repeats)?;
    scaled_dot_attention(q, &k_rep, &v_rep, scale, mask)
}

use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::attention::softmax::scaled_dot::scaled_dot_attention;
use crate::norm::qk_norm::qk_rms_norm;

pub fn qk_norm_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    weight_q: &Tensor,
    weight_k: &Tensor,
    eps: f32,
    scale: f32,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let (qn, kn) = qk_rms_norm(q, k, weight_q, weight_k, eps)?;
    scaled_dot_attention(&qn, &kn, v, scale, mask)
}

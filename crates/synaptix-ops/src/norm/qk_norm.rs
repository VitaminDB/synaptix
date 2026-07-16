use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::norm::layer_norm::layer_norm;
use crate::norm::rms_norm::rms_norm;

pub fn qk_rms_norm(
    q: &Tensor,
    k: &Tensor,
    weight_q: &Tensor,
    weight_k: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let qn = rms_norm(q, weight_q, eps)?;
    let kn = rms_norm(k, weight_k, eps)?;
    Ok((qn, kn))
}

pub fn qk_layer_norm(
    q: &Tensor,
    k: &Tensor,
    weight_q: Option<&Tensor>,
    weight_k: Option<&Tensor>,
    bias_q: Option<&Tensor>,
    bias_k: Option<&Tensor>,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let qn = layer_norm(q, weight_q, bias_q, eps)?;
    let kn = layer_norm(k, weight_k, bias_k, eps)?;
    Ok((qn, kn))
}

use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::norm::layer_norm::layer_norm;

pub fn deep_norm(
    x: &Tensor,
    residual: &Tensor,
    alpha: f32,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let res_f32 = residual.to_dtype(DType::F32)?;
    let combined = res_f32.mul_scalar(alpha)?.add(&x_f32)?;
    let normed = layer_norm(&combined, weight, bias, eps)?;
    normed.to_dtype(dtype_in)
}

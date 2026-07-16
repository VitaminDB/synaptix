use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::norm::layer_norm::layer_norm;

pub fn adaln(x: &Tensor, scale: &Tensor, shift: &Tensor, eps: f32) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let normed = layer_norm(x, None, None, eps)?.to_dtype(DType::F32)?;
    let scale_f32 = scale.to_dtype(DType::F32)?;
    let shift_f32 = shift.to_dtype(DType::F32)?;
    let scaled = normed
        .broadcast_mul(&scale_f32.add_scalar(1.0)?)?
        .broadcast_add(&shift_f32)?;
    scaled.to_dtype(dtype_in)
}

pub fn adaln_zero(
    x: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    gate: &Tensor,
    residual: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let modulated = adaln(x, scale, shift, eps)?.to_dtype(DType::F32)?;
    let g = gate.to_dtype(DType::F32)?;
    let r = residual.to_dtype(DType::F32)?;
    let gated = modulated.broadcast_mul(&g)?;
    let out = r.add(&gated)?;
    out.to_dtype(dtype_in)
}

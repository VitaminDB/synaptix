use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn elu(x: &Tensor, alpha: f32) -> Result<Tensor> {
    let x_f32 = x.to_dtype(synaptix_core::dtype::DType::F32)?;
    let pos = x_f32.clamp(0.0, f32::INFINITY)?;
    let x_clamped_neg = x_f32.clamp(f32::NEG_INFINITY, 0.0)?;
    let neg_part = x_clamped_neg.exp()?.add_scalar(-1.0)?.mul_scalar(alpha)?;
    let result_f32 = pos.add(&neg_part)?;
    result_f32.to_dtype(x.dtype())
}

pub fn hardswish(x: &Tensor) -> Result<Tensor> {
    let x_f32 = x.to_dtype(synaptix_core::dtype::DType::F32)?;
    let relu6 = x_f32.add_scalar(3.0)?.clamp(0.0, 6.0)?;
    let result_f32 = x_f32.mul(&relu6)?.mul_scalar(1.0 / 6.0)?;
    result_f32.to_dtype(x.dtype())
}

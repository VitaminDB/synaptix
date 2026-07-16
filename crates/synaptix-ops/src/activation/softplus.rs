use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn softplus(x: &Tensor, beta: f32, threshold: f32) -> Result<Tensor> {
    let _ = threshold;
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let scaled = x_f32.mul_scalar(beta)?;
    let max_part = scaled.clamp(0.0, f32::INFINITY)?;
    let abs_part = scaled.abs()?;
    let log_part = abs_part.mul_scalar(-1.0)?.exp()?.add_scalar(1.0)?.log()?;
    let inv_beta = 1.0 / beta;
    let out = max_part.add(&log_part)?.mul_scalar(inv_beta)?;
    out.to_dtype(dtype_in)
}

pub fn softsign(x: &Tensor) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let denom = x_f32.abs()?.add_scalar(1.0)?;
    let out = x_f32.div(&denom)?;
    out.to_dtype(dtype_in)
}

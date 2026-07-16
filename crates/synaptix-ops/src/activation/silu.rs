use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn silu(x: &Tensor) -> Result<Tensor> { x.silu() }

pub fn swish_beta(x: &Tensor, beta: f32) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(synaptix_core::dtype::DType::F32)?;
    let scaled = x_f32.mul_scalar(beta)?;
    let sig = scaled.sigmoid()?;
    let out = x_f32.mul(&sig)?;
    out.to_dtype(dtype_in)
}

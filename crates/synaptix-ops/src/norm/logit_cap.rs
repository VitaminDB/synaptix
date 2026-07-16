use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn soft_cap(x: &Tensor, cap: f32) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let inv = 1.0 / cap;
    let scaled = x_f32.mul_scalar(inv)?;
    let out = scaled.tanh()?.mul_scalar(cap)?;
    out.to_dtype(dtype_in)
}

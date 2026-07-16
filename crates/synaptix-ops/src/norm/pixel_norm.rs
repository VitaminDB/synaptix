use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn pixel_norm(x: &Tensor, eps: f32) -> Result<Tensor> {
    if x.rank() < 2 {
        return Err(SynaptixError::Unsupported("pixel_norm: rank must be >= 2"));
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let last = x_f32.rank() - 1;
    let mean = x_f32.sqr()?.mean_keepdim(last)?;
    let inv = mean.add_scalar(eps)?.sqrt()?.recip()?;
    let out = x_f32.broadcast_mul(&inv)?;
    out.to_dtype(dtype_in)
}

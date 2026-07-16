pub mod linear;
pub mod softmax;
pub mod tree_attn;

use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn softmax_dim(x: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = x.rank();
    if dim >= rank {
        return Err(SynaptixError::DimOutOfRange { dim, rank });
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let m = x_f32.max_keepdim(dim)?;
    let shifted = x_f32.broadcast_sub(&m)?;
    let e = shifted.exp()?;
    let s = e.sum_keepdim(dim)?;
    let out = e.broadcast_div(&s)?;
    out.to_dtype(dtype_in)
}

pub fn log_softmax_dim(x: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = x.rank();
    if dim >= rank {
        return Err(SynaptixError::DimOutOfRange { dim, rank });
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let m = x_f32.max_keepdim(dim)?;
    let shifted = x_f32.broadcast_sub(&m)?;
    let log_sum = shifted.exp()?.sum_keepdim(dim)?.log()?;
    let out = shifted.broadcast_sub(&log_sum)?;
    out.to_dtype(dtype_in)
}

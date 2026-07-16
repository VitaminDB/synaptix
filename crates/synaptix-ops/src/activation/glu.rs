use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn glu(x: &Tensor, dim: usize) -> Result<Tensor> {
    let rank = x.rank();
    if dim >= rank {
        return Err(SynaptixError::DimOutOfRange { dim, rank });
    }
    let size = x.dims()[dim];
    if size % 2 != 0 {
        return Err(SynaptixError::Unsupported("glu: dim size must be even"));
    }
    let half = size / 2;
    let a = x.narrow(dim, 0, half)?;
    let b = x.narrow(dim, half, half)?;
    let gate = b.sigmoid()?;
    a.contiguous()?.mul(&gate.contiguous()?)
}

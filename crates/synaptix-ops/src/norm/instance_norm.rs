use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn instance_norm(
    x: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    if x.rank() < 3 {
        return Err(SynaptixError::Unsupported("instance_norm: rank must be >= 3 (N, C, ...)"));
    }
    let dims = x.dims().to_vec();
    let channels = dims[1];
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let reduce_dims: Vec<usize> = (2..dims.len()).collect();
    let spatial: usize = reduce_dims.iter().map(|&d| dims[d]).product();
    let sum = x_f32.sum(&reduce_dims[..])?;
    let mean = sum.mul_scalar(1.0 / (spatial as f32))?;
    let mut keepdim_shape = vec![1usize; dims.len()];
    keepdim_shape[0] = dims[0];
    keepdim_shape[1] = dims[1];
    let mean_kd = mean.reshape(keepdim_shape.clone())?;
    let centered = x_f32.broadcast_sub(&mean_kd)?;
    let var_sum = centered.sqr()?.sum(&reduce_dims[..])?;
    let var_kd = var_sum.mul_scalar(1.0 / (spatial as f32))?.reshape(keepdim_shape)?;
    let inv = var_kd.add_scalar(eps)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv)?;
    let scaled = match weight {
        Some(w) => {
            let mut w_shape = vec![1usize; dims.len()];
            w_shape[1] = channels;
            let w_view = w.to_dtype(DType::F32)?.reshape(w_shape)?;
            normed.broadcast_mul(&w_view)?
        }
        None => normed,
    };
    let out = match bias {
        Some(b) => {
            let mut b_shape = vec![1usize; dims.len()];
            b_shape[1] = channels;
            let b_view = b.to_dtype(DType::F32)?.reshape(b_shape)?;
            scaled.broadcast_add(&b_view)?
        }
        None => scaled,
    };
    out.to_dtype(dtype_in)
}

#[derive(Debug, Clone)]
pub struct InstanceNorm {
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
}

impl InstanceNorm {
    pub fn new(weight: Option<Tensor>, bias: Option<Tensor>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        instance_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.eps)
    }
}

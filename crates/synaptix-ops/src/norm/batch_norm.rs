use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn batch_norm_inference(
    x: &Tensor,
    running_mean: &Tensor,
    running_var: &Tensor,
    weight: Option<&Tensor>,
    bias: Option<&Tensor>,
    eps: f32,
) -> Result<Tensor> {
    if x.rank() < 2 {
        return Err(SynaptixError::Unsupported("batch_norm: rank must be >= 2"));
    }
    let dims = x.dims().to_vec();
    let channels = dims[1];
    let mut chan_shape = vec![1usize; dims.len()];
    chan_shape[1] = channels;
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let mean = running_mean.to_dtype(DType::F32)?.reshape(chan_shape.clone())?;
    let var = running_var.to_dtype(DType::F32)?.reshape(chan_shape.clone())?;
    let inv = var.add_scalar(eps)?.sqrt()?.recip()?;
    let normed = x_f32.broadcast_sub(&mean)?.broadcast_mul(&inv)?;
    let scaled = match weight {
        Some(w) => {
            let w_view = w.to_dtype(DType::F32)?.reshape(chan_shape.clone())?;
            normed.broadcast_mul(&w_view)?
        }
        None => normed,
    };
    let out = match bias {
        Some(b) => {
            let b_view = b.to_dtype(DType::F32)?.reshape(chan_shape)?;
            scaled.broadcast_add(&b_view)?
        }
        None => scaled,
    };
    out.to_dtype(dtype_in)
}

#[derive(Debug, Clone)]
pub struct BatchNorm {
    running_mean: Tensor,
    running_var: Tensor,
    weight: Option<Tensor>,
    bias: Option<Tensor>,
    eps: f32,
}

impl BatchNorm {
    pub fn new(
        running_mean: Tensor,
        running_var: Tensor,
        weight: Option<Tensor>,
        bias: Option<Tensor>,
        eps: f32,
    ) -> Self {
        Self { running_mean, running_var, weight, bias, eps }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        batch_norm_inference(
            x,
            &self.running_mean,
            &self.running_var,
            self.weight.as_ref(),
            self.bias.as_ref(),
            self.eps,
        )
    }
}

use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

pub fn relu(x: &Tensor) -> Result<Tensor> {
    let zero = Tensor::zeros(vec![1; x.rank()], x.dtype(), x.device())?
        .broadcast_as(x.dims().to_vec())?;
    x.maximum(&zero)
}

pub fn relu_squared(x: &Tensor) -> Result<Tensor> {
    let r = relu(x)?;
    r.sqr()
}

pub fn leaky_relu(x: &Tensor, negative_slope: f32) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let scaled = x_f32.mul_scalar(negative_slope)?;
    let out = x_f32.maximum(&scaled)?;
    out.to_dtype(dtype_in)
}

pub fn prelu(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if x.device() != weight.device() {
        return Err(SynaptixError::device_mismatch(x.device(), weight.device()));
    }
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let mut w_shape = vec![1usize; x_f32.rank()];
    if weight.numel() == 1 {
    } else if x_f32.rank() >= 2 && weight.rank() == 1 && weight.dims()[0] == x_f32.dims()[1] {
        w_shape[1] = weight.dims()[0];
    } else if weight.dims() == x_f32.dims() {
        w_shape = x_f32.dims().to_vec();
    } else {
        return Err(SynaptixError::shape_mismatch(x_f32.dims(), weight.dims()));
    }
    let w_view = w_f32.reshape(w_shape)?;
    let w_broadcast = w_view.broadcast_as(x_f32.dims().to_vec())?;
    let zero = Tensor::zeros(x_f32.dims().to_vec(), DType::F32, x_f32.device())?;
    let pos = x_f32.maximum(&zero)?;
    let neg = x_f32.minimum(&zero)?.mul(&w_broadcast)?;
    let out = pos.add(&neg)?;
    out.to_dtype(dtype_in)
}

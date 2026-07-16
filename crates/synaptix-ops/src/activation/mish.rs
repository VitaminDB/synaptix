use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::activation::softplus::softplus;

pub fn mish(x: &Tensor) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let sp = softplus(&x_f32, 1.0, 20.0)?;
    let t = sp.tanh()?;
    let out = x_f32.mul(&t)?;
    out.to_dtype(dtype_in)
}

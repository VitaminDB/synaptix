use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn gelu_tanh(x: &Tensor) -> Result<Tensor> { x.gelu_tanh() }

pub fn gelu_exact(x: &Tensor) -> Result<Tensor> { x.gelu_exact() }

pub fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let s = x_f32.mul_scalar(1.702)?;
    let sig = s.sigmoid()?;
    let out = x_f32.mul(&sig)?;
    out.to_dtype(dtype_in)
}

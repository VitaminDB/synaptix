use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn dyn_tanh(x: &Tensor, gamma: &Tensor, beta: &Tensor) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let g_f32 = gamma.to_dtype(DType::F32)?;
    let b_f32 = beta.to_dtype(DType::F32)?;
    let scaled = x_f32.broadcast_mul(&g_f32)?.broadcast_add(&b_f32)?;
    let out = scaled.tanh()?;
    out.to_dtype(dtype_in)
}

pub fn dyn_tanh_scalar(x: &Tensor, gamma: f32, beta: f32) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let out = x_f32.affine(gamma, beta)?.tanh()?;
    out.to_dtype(dtype_in)
}

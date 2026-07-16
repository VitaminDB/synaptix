use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub fn linear(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let out = x.matmul(weight)?;
    match bias {
        Some(b) => out.broadcast_add(b),
        None => Ok(out),
    }
}

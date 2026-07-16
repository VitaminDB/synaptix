use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

use super::Reduction;

fn reduce(loss: Tensor, reduction: Reduction) -> Result<Tensor> {
    match reduction {
        Reduction::None => Ok(loss),
        Reduction::Sum => loss.flatten_all()?.sum_all(),
        Reduction::Mean => {
            let n = loss.shape().numel().max(1) as f32;
            let s = loss.flatten_all()?.sum_all()?;
            s.affine(1.0 / n, 0.0)
        }
    }
}

pub fn mse_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    if input.dims() != target.dims() {
        return Err(SynaptixError::shape_mismatch(input.dims(), target.dims()));
    }
    let input_f = input.to_dtype(DType::F32)?;
    let target_f = target.to_dtype(DType::F32)?;
    let diff = input_f.sub(&target_f)?;
    let sq = diff.sqr()?;
    reduce(sq, reduction)
}

pub fn l1_loss(input: &Tensor, target: &Tensor, reduction: Reduction) -> Result<Tensor> {
    if input.dims() != target.dims() {
        return Err(SynaptixError::shape_mismatch(input.dims(), target.dims()));
    }
    let input_f = input.to_dtype(DType::F32)?;
    let target_f = target.to_dtype(DType::F32)?;
    let diff = input_f.sub(&target_f)?.abs()?;
    reduce(diff, reduction)
}

pub fn smooth_l1_loss(input: &Tensor, target: &Tensor, beta: f32, reduction: Reduction) -> Result<Tensor> {
    if input.dims() != target.dims() {
        return Err(SynaptixError::shape_mismatch(input.dims(), target.dims()));
    }
    let input_f = input.to_dtype(DType::F32)?;
    let target_f = target.to_dtype(DType::F32)?;
    let diff = input_f.sub(&target_f)?;
    let abs = diff.abs()?;
    let flat = abs.flatten_all()?.to_vec1::<f32>()?;
    let losses: Vec<f32> = flat
        .iter()
        .map(|&a| {
            if a < beta {
                0.5 * a * a / beta
            } else {
                a - 0.5 * beta
            }
        })
        .collect();
    let loss = Tensor::from_vec::<_, f32>(losses, input.dims().to_vec(), input.device())?;
    reduce(loss, reduction)
}

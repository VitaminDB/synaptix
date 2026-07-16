use synaptix_core::dtype::DType;
use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;
use synaptix_ops::attention::log_softmax_dim;

use super::Reduction;

pub fn cross_entropy(
    logits: &Tensor,
    target_ids: &Tensor,
    ignore_index: Option<i64>,
    reduction: Reduction,
) -> Result<Tensor> {
    if logits.rank() < 2 {
        return Err(SynaptixError::Unsupported(
            "cross_entropy: logits rank must be >= 2 (..., vocab)",
        ));
    }
    let vocab_dim = logits.rank() - 1;
    let vocab = logits.dims()[vocab_dim];
    let logits_f = logits.to_dtype(DType::F32)?;
    let log_probs = log_softmax_dim(&logits_f, vocab_dim)?;

    let target_i = target_ids.to_dtype(DType::I64)?.contiguous()?;
    let target_flat = target_i.flatten_all()?.to_vec1::<i64>()?;
    let log_probs_flat = log_probs
        .reshape(vec![target_flat.len(), vocab])?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut losses = Vec::with_capacity(target_flat.len());
    let mut n_valid = 0usize;
    for (i, &tgt) in target_flat.iter().enumerate() {
        if let Some(ign) = ignore_index {
            if tgt == ign {
                losses.push(0.0_f32);
                continue;
            }
        }
        if tgt < 0 || tgt as usize >= vocab {
            return Err(SynaptixError::Other(format!(
                "cross_entropy: target {} out of range [0, {})",
                tgt, vocab
            )));
        }
        let lp = log_probs_flat[i * vocab + tgt as usize];
        losses.push(-lp);
        n_valid += 1;
    }

    let target_shape = target_ids.dims().to_vec();
    let loss_tensor = Tensor::from_vec::<_, f32>(losses, target_shape, logits.device())?;
    match reduction {
        Reduction::None => Ok(loss_tensor),
        Reduction::Sum => {
            let s = loss_tensor.flatten_all()?.sum_all()?;
            Ok(s)
        }
        Reduction::Mean => {
            let s = loss_tensor.flatten_all()?.sum_all()?;
            let denom = n_valid.max(1) as f32;
            s.affine(1.0 / denom, 0.0)
        }
    }
}

pub fn nll_loss(
    log_probs: &Tensor,
    target_ids: &Tensor,
    ignore_index: Option<i64>,
    reduction: Reduction,
) -> Result<Tensor> {
    if log_probs.rank() < 2 {
        return Err(SynaptixError::Unsupported(
            "nll_loss: log_probs rank must be >= 2 (..., vocab)",
        ));
    }
    let vocab_dim = log_probs.rank() - 1;
    let vocab = log_probs.dims()[vocab_dim];
    let log_probs_f = log_probs.to_dtype(DType::F32)?;

    let target_i = target_ids.to_dtype(DType::I64)?.contiguous()?;
    let target_flat = target_i.flatten_all()?.to_vec1::<i64>()?;
    let lp_flat = log_probs_f
        .reshape(vec![target_flat.len(), vocab])?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut losses = Vec::with_capacity(target_flat.len());
    let mut n_valid = 0usize;
    for (i, &tgt) in target_flat.iter().enumerate() {
        if let Some(ign) = ignore_index {
            if tgt == ign {
                losses.push(0.0_f32);
                continue;
            }
        }
        if tgt < 0 || tgt as usize >= vocab {
            return Err(SynaptixError::Other(format!(
                "nll_loss: target {} out of range [0, {})",
                tgt, vocab
            )));
        }
        losses.push(-lp_flat[i * vocab + tgt as usize]);
        n_valid += 1;
    }

    let loss_tensor = Tensor::from_vec::<_, f32>(losses, target_ids.dims().to_vec(), log_probs.device())?;
    match reduction {
        Reduction::None => Ok(loss_tensor),
        Reduction::Sum => loss_tensor.flatten_all()?.sum_all(),
        Reduction::Mean => {
            let s = loss_tensor.flatten_all()?.sum_all()?;
            s.affine(1.0 / n_valid.max(1) as f32, 0.0)
        }
    }
}

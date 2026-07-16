use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

use crate::functional::activation::log_softmax;

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    pred.sub(target)?.sqr()?.mean()
}

pub fn mae_loss(pred: &Tensor, target: &Tensor) -> Result<Tensor> {
    pred.sub(target)?.abs()?.mean()
}

pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Result<Tensor> {
    let diff = pred.sub(target)?;
    let abs = diff.abs()?;
    let quad = diff.sqr()?.affine(0.5, 0.0)?;
    let mask_lin = abs.add_scalar(-delta)?.step_gt_zero()?;
    let mask_quad = mask_lin.affine(-1.0, 1.0)?;
    let linear_part = abs.affine(delta, -0.5 * delta * delta)?;
    quad.mul(&mask_quad)?.add(&linear_part.mul(&mask_lin)?)?.mean()
}

pub fn bce_with_logits(logits: &Tensor, target: &Tensor) -> Result<Tensor> {
    let abs_x = logits.abs()?;
    let log_term = abs_x.neg()?.exp()?.add_scalar(1.0)?.log()?;
    let xy = logits.mul(target)?;
    let max_x_0 = logits.relu()?;
    max_x_0.sub(&xy)?.add(&log_term)?.mean()
}

pub fn cross_entropy_with_one_hot(logits: &Tensor, target_one_hot: &Tensor, dim: usize) -> Result<Tensor> {
    let log_sm = log_softmax(logits, dim)?;
    let weighted = log_sm.mul(target_one_hot)?;
    let summed = weighted.sum(vec![dim])?;
    summed.mean()?.neg()
}

pub fn kl_div(log_pred: &Tensor, target_dist: &Tensor) -> Result<Tensor> {
    let log_target = target_dist.add_scalar(1e-12)?.log()?;
    let diff = log_target.sub(log_pred)?;
    let term = target_dist.mul(&diff)?;
    term.mean()
}

pub fn focal_loss_with_logits(logits: &Tensor, target: &Tensor, gamma: f32) -> Result<Tensor> {
    let p = logits.sigmoid()?;
    let one_minus_p = p.affine(-1.0, 1.0)?;
    let one_minus_target = target.affine(-1.0, 1.0)?;
    let pt = p.mul(target)?.add(&one_minus_p.mul(&one_minus_target)?)?;
    let log_pt = pt.add_scalar(1e-12)?.log()?;
    let modulating = pt.affine(-1.0, 1.0)?.powf(gamma)?;
    modulating.mul(&log_pt)?.mean()?.neg()
}

pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
) -> Result<Tensor> {
    let d_pos = anchor.sub(positive)?.sqr()?.sum(vec![1])?;
    let d_neg = anchor.sub(negative)?.sqr()?.sum(vec![1])?;
    let diff = d_pos.sub(&d_neg)?.add_scalar(margin)?;
    diff.relu()?.mean()
}

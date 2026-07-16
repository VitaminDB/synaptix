use std::sync::Arc;

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::grad::{GradFn, UnaryGradKind};
use synaptix_core::tensor::Tensor;

pub struct UnaryGradFn {
    parents: [Tensor; 1],
    saved_input: Tensor,
    kind: UnaryGradKind,
    alpha: Option<f32>,
}

impl UnaryGradFn {
    pub fn new(input: &Tensor, kind: UnaryGradKind, alpha: Option<f32>) -> Arc<dyn GradFn> {
        Arc::new(Self {
            saved_input: input.detach(),
            parents: [input.clone()],
            kind,
            alpha,
        })
    }
}

impl GradFn for UnaryGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let x = &self.saved_input;
        let dydx = match self.kind {
            UnaryGradKind::Sigmoid => {
                let s = x.sigmoid()?;
                let one_minus_s = s.affine(-1.0, 1.0)?;
                s.mul(&one_minus_s)?
            }
            UnaryGradKind::Tanh => {
                let t = x.tanh()?;
                let t_sq = t.mul(&t)?;
                t_sq.affine(-1.0, 1.0)?
            }
            UnaryGradKind::SiLU => {
                let s = x.sigmoid()?;
                let one_minus_s = s.affine(-1.0, 1.0)?;
                let part2 = x.mul(&s)?.mul(&one_minus_s)?;
                s.add(&part2)?
            }
            UnaryGradKind::Erf => {
                let x_sq = x.mul(x)?;
                let neg = x_sq.neg()?;
                let e = neg.exp()?;
                let coef = 2.0f32 / std::f32::consts::PI.sqrt();
                e.affine(coef, 0.0)?
            }
            UnaryGradKind::GeLUTanh => {
                let c = (2.0f32 / std::f32::consts::PI).sqrt();
                let x_sq = x.mul(x)?;
                let x_cubed = x_sq.mul(x)?;
                let u = x.add(&x_cubed.affine(0.044715, 0.0)?)?.affine(c, 0.0)?;
                let tanh_u = u.tanh()?;
                let one_plus_tanh = tanh_u.affine(1.0, 1.0)?;
                let term1 = one_plus_tanh.affine(0.5, 0.0)?;
                let one_minus_tanh_sq = tanh_u.mul(&tanh_u)?.affine(-1.0, 1.0)?;
                let du_dx = x_sq.affine(3.0 * 0.044715, 1.0)?.affine(c, 0.0)?;
                let term2 = x.affine(0.5, 0.0)?.mul(&one_minus_tanh_sq)?.mul(&du_dx)?;
                term1.add(&term2)?
            }
            UnaryGradKind::GeLUExact => {
                let x_over_sqrt2 = x.affine(1.0 / std::f32::consts::SQRT_2, 0.0)?;
                let erf_part = x_over_sqrt2.erf()?;
                let part1 = erf_part.affine(0.5, 0.5)?;
                let x_sq = x.mul(x)?;
                let neg_half = x_sq.affine(-0.5, 0.0)?;
                let exp_part = neg_half.exp()?;
                let coef = 1.0f32 / (2.0 * std::f32::consts::PI).sqrt();
                let part2 = x.mul(&exp_part)?.affine(coef, 0.0)?;
                part1.add(&part2)?
            }
            UnaryGradKind::Exp => x.exp()?,
            UnaryGradKind::Log => x.recip()?,
            UnaryGradKind::Recip => {
                let r = x.recip()?;
                r.mul(&r)?.neg()?
            }
            UnaryGradKind::Sqrt => {
                let s = x.sqrt()?;
                s.recip()?.affine(0.5, 0.0)?
            }
            UnaryGradKind::Square => x.affine(2.0, 0.0)?,
            UnaryGradKind::Neg => {
                let ones = Tensor::ones(x.shape().clone(), x.dtype(), x.device())?;
                ones.neg()?
            }
            UnaryGradKind::Abs => x.sign()?,
            UnaryGradKind::Relu => x.step_gt_zero()?,
            UnaryGradKind::Relu2 => {
                let mask = x.step_gt_zero()?;
                x.mul(&mask)?.affine(2.0, 0.0)?
            }
            UnaryGradKind::LeakyRelu => {
                let alpha = match self.alpha {
                    Some(a) => a,
                    None => {
                        return Err(SynaptixError::Other(
                            "LeakyRelu backward requires alpha".into(),
                        ));
                    }
                };
                let mask = x.step_gt_zero()?;
                let neg_mask = mask.affine(-1.0, 1.0)?;
                let pos_part = mask;
                let neg_part = neg_mask.affine(alpha, 0.0)?;
                pos_part.add(&neg_part)?
            }
            UnaryGradKind::Sign => {
                let zeros = Tensor::zeros(x.shape().clone(), x.dtype(), x.device())?;
                zeros
            }
            UnaryGradKind::StepGtZero => {
                let zeros = Tensor::zeros(x.shape().clone(), x.dtype(), x.device())?;
                zeros
            }
            UnaryGradKind::Rsqrt => {
                let s = x.sqrt()?;
                let s3 = s.mul(&s)?.mul(&s)?;
                s3.recip()?.affine(-0.5, 0.0)?
            }
            other => {
                return Err(SynaptixError::Other(format!(
                    "UnaryGradFn backward not yet implemented for {:?}",
                    other
                )));
            }
        };
        Ok(vec![Some(output_grad.mul(&dydx)?)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "UnaryGradFn"
    }
}

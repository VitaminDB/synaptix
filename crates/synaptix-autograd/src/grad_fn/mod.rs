pub mod activation;
pub mod arithmetic;
pub mod attention;
pub mod cat;
pub mod conv;
pub mod indexing;
pub mod matmul;
pub mod norm;
pub mod reduction;
pub mod reshape;
pub mod util;

use std::sync::Arc;

use synaptix_core::grad::{GradFn, GradFnBuilder, GradOp};
use synaptix_core::tensor::Tensor;

pub struct Builder;

impl GradFnBuilder for Builder {
    fn build(&self, op: GradOp<'_>, output: &Tensor) -> Option<Arc<dyn GradFn>> {
        let _ = output;
        match op {
            GradOp::Identity { input } => Some(reshape::IdentityGradFn::new(input)),
            GradOp::Add { lhs, rhs } => Some(arithmetic::AddGradFn::new(lhs, rhs)),
            GradOp::Sub { lhs, rhs } => Some(arithmetic::SubGradFn::new(lhs, rhs)),
            GradOp::Mul { lhs, rhs } => Some(arithmetic::MulGradFn::new(lhs, rhs)),
            GradOp::Div { lhs, rhs } => Some(arithmetic::DivGradFn::new(lhs, rhs)),
            GradOp::Neg { input } => Some(arithmetic::NegGradFn::new(input)),
            GradOp::Affine { input, mul, add } => {
                Some(arithmetic::AffineGradFn::new(input, mul, add))
            }
            GradOp::AddScalar { input, .. } => {
                Some(arithmetic::AffineGradFn::new(input, 1.0, 0.0))
            }
            GradOp::MulScalar { input, scalar } => {
                Some(arithmetic::AffineGradFn::new(input, scalar, 0.0))
            }
            GradOp::Sum { input, dims, keepdim } => {
                Some(reduction::SumGradFn::new(input, dims, keepdim))
            }
            GradOp::Mean { input, dims, keepdim } => {
                Some(reduction::MeanGradFn::new(input, dims, keepdim))
            }
            GradOp::MatMul { lhs, rhs } => Some(matmul::MatMulGradFn::new(lhs, rhs)),
            GradOp::Cast { input, target_dtype } => {
                Some(reshape::CastGradFn::new(input, target_dtype))
            }
            GradOp::Reshape { input } => Some(reshape::ReshapeGradFn::new(input)),
            GradOp::Transpose { input, dim0, dim1 } => {
                Some(reshape::TransposeGradFn::new(input, dim0, dim1))
            }
            GradOp::Permute { input, perm } => {
                Some(reshape::PermuteGradFn::new(input, perm))
            }
            GradOp::Squeeze { input, dim } => {
                Some(reshape::SqueezeGradFn::new(input, dim))
            }
            GradOp::Unsqueeze { input, dim } => {
                Some(reshape::UnsqueezeGradFn::new(input, dim))
            }
            GradOp::Unary { input, kind, alpha } => {
                Some(activation::UnaryGradFn::new(input, kind, alpha))
            }
            GradOp::Gather { input, indices, dim } => {
                Some(indexing::GatherGradFn::new(input, indices, dim))
            }
            GradOp::IndexSelect { input, indices, dim } => {
                Some(indexing::IndexSelectGradFn::new(input, indices, dim))
            }
            GradOp::MaskedFill { input, mask, .. } => {
                Some(indexing::MaskedFillGradFn::new(input, mask))
            }
            GradOp::WhereCond { cond, a, b } => {
                Some(indexing::WhereCondGradFn::new(cond, a, b))
            }
            GradOp::Cat { inputs, dim } => {
                let refs: Vec<&Tensor> = inputs.iter().copied().collect();
                Some(cat::CatGradFn::new(&refs, dim))
            }
            _ => None,
        }
    }
}

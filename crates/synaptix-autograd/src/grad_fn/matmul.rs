use std::sync::Arc;

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::grad::GradFn;
use synaptix_core::tensor::Tensor;

use crate::grad_fn::util::unbroadcast_to;

pub struct MatMulGradFn {
    parents: [Tensor; 2],
    saved: [Tensor; 2],
    shapes: [Vec<usize>; 2],
}

impl MatMulGradFn {
    pub fn new(lhs: &Tensor, rhs: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            shapes: [lhs.dims().to_vec(), rhs.dims().to_vec()],
            saved: [lhs.detach(), rhs.detach()],
            parents: [lhs.clone(), rhs.clone()],
        })
    }
}

impl GradFn for MatMulGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let a = &self.saved[0];
        let b = &self.saved[1];
        let a_rank = a.rank();
        let b_rank = b.rank();
        if a_rank < 2 || b_rank < 2 {
            return Err(SynaptixError::RankMismatch { expected: 2, got: a_rank.min(b_rank) });
        }
        let b_t = b.transpose(b_rank - 2, b_rank - 1)?.contiguous()?;
        let a_t = a.transpose(a_rank - 2, a_rank - 1)?.contiguous()?;

        let da_full = output_grad.matmul(&b_t)?;
        let db_full = a_t.matmul(output_grad)?;

        let da = unbroadcast_to(da_full, &self.shapes[0])?;
        let db = unbroadcast_to(db_full, &self.shapes[1])?;
        Ok(vec![Some(da), Some(db)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "MatMulGradFn"
    }
}

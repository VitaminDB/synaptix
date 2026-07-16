use crate::error::Result;
use crate::grad::{self, GradOp};
use crate::tensor::Tensor;
use crate::tensor::shape::IntoShape;

impl Tensor {
    pub fn reshape<S: IntoShape>(&self, shape: S) -> Result<Self> {
        let new_shape = shape.into_shape();
        let layout = self.layout.reshape(new_shape)?;
        let mut out = self.with_layout(layout);
        grad::try_attach_grad_fn(GradOp::Reshape { input: self }, &mut out)?;
        Ok(out)
    }

    pub fn transpose(&self, d1: usize, d2: usize) -> Result<Self> {
        let layout = self.layout.transpose(d1, d2)?;
        let mut out = self.with_layout(layout);
        grad::try_attach_grad_fn(GradOp::Transpose { input: self, dim0: d1, dim1: d2 }, &mut out)?;
        Ok(out)
    }

    pub fn t(&self) -> Result<Self> {
        let rank = self.rank();
        if rank < 2 {
            return Err(crate::error::SynaptixError::RankMismatch { expected: 2, got: rank });
        }
        self.transpose(rank - 2, rank - 1)
    }

    pub fn permute<P: AsRef<[usize]>>(&self, perm: P) -> Result<Self> {
        let perm_vec: Vec<usize> = perm.as_ref().to_vec();
        let layout = self.layout.permute(&perm_vec)?;
        let mut out = self.with_layout(layout);
        grad::try_attach_grad_fn(GradOp::Permute { input: self, perm: perm_vec }, &mut out)?;
        Ok(out)
    }

    pub fn squeeze(&self, dim: usize) -> Result<Self> {
        let layout = self.layout.squeeze(dim)?;
        let mut out = self.with_layout(layout);
        grad::try_attach_grad_fn(GradOp::Squeeze { input: self, dim }, &mut out)?;
        Ok(out)
    }

    pub fn flatten_all(&self) -> Result<Self> {
        let layout = self.layout.flatten_all()?;
        Ok(self.with_layout(layout))
    }
}

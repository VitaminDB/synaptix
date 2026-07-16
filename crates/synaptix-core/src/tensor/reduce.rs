use crate::backend::ReduceOp;
use crate::error::Result;
use crate::tensor::Tensor;
use crate::tensor::ops::run_reduce;

impl Tensor {
    pub fn sum_all(&self) -> Result<Self> {
        let dims: Vec<usize> = (0..self.rank()).collect();
        run_reduce(self, ReduceOp::Sum, &dims, false)
    }

    pub fn sum<D: AsRef<[usize]>>(&self, dims: D) -> Result<Self> {
        run_reduce(self, ReduceOp::Sum, dims.as_ref(), false)
    }

    pub fn sum_keepdim(&self, dim: usize) -> Result<Self> {
        run_reduce(self, ReduceOp::Sum, &[dim], true)
    }

    pub fn mean(&self) -> Result<Self> {
        let dims: Vec<usize> = (0..self.rank()).collect();
        run_reduce(self, ReduceOp::Mean, &dims, false)
    }

    pub fn mean_keepdim(&self, dim: usize) -> Result<Self> {
        run_reduce(self, ReduceOp::Mean, &[dim], true)
    }

    pub fn max_all(&self) -> Result<Self> {
        let dims: Vec<usize> = (0..self.rank()).collect();
        run_reduce(self, ReduceOp::Max, &dims, false)
    }

    pub fn max<D: AsRef<[usize]>>(&self, dims: D) -> Result<Self> {
        run_reduce(self, ReduceOp::Max, dims.as_ref(), false)
    }

    pub fn max_keepdim(&self, dim: usize) -> Result<Self> {
        run_reduce(self, ReduceOp::Max, &[dim], true)
    }

    pub fn argmax(&self, dim: usize) -> Result<Self> {
        run_reduce(self, ReduceOp::ArgMax, &[dim], false)
    }
}

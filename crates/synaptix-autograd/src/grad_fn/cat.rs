use std::sync::Arc;

use synaptix_core::error::Result;
use synaptix_core::grad::GradFn;
use synaptix_core::tensor::Tensor;

pub struct CatGradFn {
    parents: Vec<Tensor>,
    dim_sizes: Vec<usize>,
    dim: usize,
}

impl CatGradFn {
    pub fn new(inputs: &[&Tensor], dim: usize) -> Arc<dyn GradFn> {
        let parents: Vec<Tensor> = inputs.iter().map(|t| (*t).clone()).collect();
        let dim_sizes: Vec<usize> = inputs.iter().map(|t| t.dims()[dim]).collect();
        Arc::new(Self { parents, dim_sizes, dim })
    }
}

impl GradFn for CatGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let mut grads = Vec::with_capacity(self.parents.len());
        let mut offset = 0usize;
        for &len in &self.dim_sizes {
            let slice = output_grad.narrow(self.dim, offset, len)?.contiguous()?;
            grads.push(Some(slice));
            offset += len;
        }
        Ok(grads)
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "CatGradFn"
    }
}

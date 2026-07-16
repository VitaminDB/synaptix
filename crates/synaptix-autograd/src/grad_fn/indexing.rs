use std::sync::Arc;

use synaptix_core::error::Result;
use synaptix_core::grad::GradFn;
use synaptix_core::tensor::Tensor;

pub struct GatherGradFn {
    parents: [Tensor; 1],
    indices: Tensor,
    dim: usize,
    input_shape: Vec<usize>,
}

impl GatherGradFn {
    pub fn new(input: &Tensor, indices: &Tensor, dim: usize) -> Arc<dyn GradFn> {
        Arc::new(Self {
            parents: [input.clone()],
            indices: indices.detach(),
            dim,
            input_shape: input.dims().to_vec(),
        })
    }
}

impl GradFn for GatherGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let zero = Tensor::zeros(
            self.input_shape.clone(),
            output_grad.dtype(),
            output_grad.device(),
        )?;
        let g_contig = output_grad.contiguous()?;
        let result = zero.scatter_add(self.dim, &self.indices, &g_contig)?;
        Ok(vec![Some(result)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "GatherGradFn"
    }
}

pub struct IndexSelectGradFn {
    parents: [Tensor; 1],
    indices: Tensor,
    dim: usize,
    input_shape: Vec<usize>,
}

impl IndexSelectGradFn {
    pub fn new(input: &Tensor, indices: &Tensor, dim: usize) -> Arc<dyn GradFn> {
        Arc::new(Self {
            parents: [input.clone()],
            indices: indices.detach(),
            dim,
            input_shape: input.dims().to_vec(),
        })
    }
}

impl GradFn for IndexSelectGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let rank = output_grad.rank();
        let idx_numel = self.indices.numel();
        let mut idx_shape = vec![1usize; rank];
        if self.dim < rank {
            idx_shape[self.dim] = idx_numel;
        }
        let indices_reshape = self.indices.reshape(idx_shape)?;
        let indices_expanded = indices_reshape
            .broadcast_as(output_grad.shape().clone())?
            .contiguous()?;
        let g_contig = output_grad.contiguous()?;
        let zero = Tensor::zeros(
            self.input_shape.clone(),
            output_grad.dtype(),
            output_grad.device(),
        )?;
        let result = zero.scatter_add(self.dim, &indices_expanded, &g_contig)?;
        Ok(vec![Some(result)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "IndexSelectGradFn"
    }
}

pub struct MaskedFillGradFn {
    parents: [Tensor; 1],
    mask: Tensor,
    input_shape: Vec<usize>,
}

impl MaskedFillGradFn {
    pub fn new(input: &Tensor, mask: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            parents: [input.clone()],
            mask: mask.detach(),
            input_shape: input.dims().to_vec(),
        })
    }
}

impl GradFn for MaskedFillGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let zero = Tensor::zeros(
            output_grad.shape().clone(),
            output_grad.dtype(),
            output_grad.device(),
        )?;
        let result = Tensor::where_cond(&self.mask, &zero, output_grad)?;
        let unbroadcasted = crate::grad_fn::util::unbroadcast_to(result, &self.input_shape)?;
        Ok(vec![Some(unbroadcasted)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "MaskedFillGradFn"
    }
}

pub struct WhereCondGradFn {
    parents: [Tensor; 2],
    cond: Tensor,
    a_shape: Vec<usize>,
    b_shape: Vec<usize>,
}

impl WhereCondGradFn {
    pub fn new(cond: &Tensor, a: &Tensor, b: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            parents: [a.clone(), b.clone()],
            cond: cond.detach(),
            a_shape: a.dims().to_vec(),
            b_shape: b.dims().to_vec(),
        })
    }
}

impl GradFn for WhereCondGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let zero = Tensor::zeros(
            output_grad.shape().clone(),
            output_grad.dtype(),
            output_grad.device(),
        )?;
        let da_full = Tensor::where_cond(&self.cond, output_grad, &zero)?;
        let db_full = Tensor::where_cond(&self.cond, &zero, output_grad)?;
        let da = crate::grad_fn::util::unbroadcast_to(da_full, &self.a_shape)?;
        let db = crate::grad_fn::util::unbroadcast_to(db_full, &self.b_shape)?;
        Ok(vec![Some(da), Some(db)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "WhereCondGradFn"
    }
}

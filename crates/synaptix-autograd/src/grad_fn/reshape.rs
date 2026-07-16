use std::sync::Arc;

use synaptix_core::dtype::DType;
use synaptix_core::error::Result;
use synaptix_core::grad::GradFn;
use synaptix_core::tensor::Tensor;

pub struct ReshapeGradFn {
    parents: [Tensor; 1],
    input_shape: Vec<usize>,
}

impl ReshapeGradFn {
    pub fn new(input: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()], input_shape: input.dims().to_vec() })
    }
}

impl GradFn for ReshapeGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.contiguous()?.reshape(self.input_shape.clone())?;
        Ok(vec![Some(g)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "ReshapeGradFn"
    }
}

pub struct TransposeGradFn {
    parents: [Tensor; 1],
    dim0: usize,
    dim1: usize,
}

impl TransposeGradFn {
    pub fn new(input: &Tensor, dim0: usize, dim1: usize) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()], dim0, dim1 })
    }
}

impl GradFn for TransposeGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.transpose(self.dim0, self.dim1)?;
        Ok(vec![Some(g)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "TransposeGradFn"
    }
}

pub struct PermuteGradFn {
    parents: [Tensor; 1],
    inverse: Vec<usize>,
}

impl PermuteGradFn {
    pub fn new(input: &Tensor, perm: Vec<usize>) -> Arc<dyn GradFn> {
        let mut inverse = vec![0usize; perm.len()];
        for (i, &p) in perm.iter().enumerate() {
            inverse[p] = i;
        }
        Arc::new(Self { parents: [input.clone()], inverse })
    }
}

impl GradFn for PermuteGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.permute(self.inverse.clone())?;
        Ok(vec![Some(g)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "PermuteGradFn"
    }
}

pub struct SqueezeGradFn {
    parents: [Tensor; 1],
    dim: usize,
}

impl SqueezeGradFn {
    pub fn new(input: &Tensor, dim: usize) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()], dim })
    }
}

impl GradFn for SqueezeGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.unsqueeze(self.dim)?;
        Ok(vec![Some(g)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "SqueezeGradFn"
    }
}

pub struct UnsqueezeGradFn {
    parents: [Tensor; 1],
    dim: usize,
}

impl UnsqueezeGradFn {
    pub fn new(input: &Tensor, dim: usize) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()], dim })
    }
}

impl GradFn for UnsqueezeGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.squeeze(self.dim)?;
        Ok(vec![Some(g)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "UnsqueezeGradFn"
    }
}

pub struct IdentityGradFn {
    parents: [Tensor; 1],
}

impl IdentityGradFn {
    pub fn new(input: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()] })
    }
}

impl GradFn for IdentityGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(output_grad.clone())])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "IdentityGradFn"
    }
}

pub struct CastGradFn {
    parents: [Tensor; 1],
    source_dtype: DType,
}

impl CastGradFn {
    pub fn new(input: &Tensor, _target_dtype: DType) -> Arc<dyn GradFn> {
        Arc::new(Self { source_dtype: input.dtype(), parents: [input.clone()] })
    }
}

impl GradFn for CastGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let g = output_grad.to_dtype(self.source_dtype)?;
        Ok(vec![Some(g)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "CastGradFn"
    }
}

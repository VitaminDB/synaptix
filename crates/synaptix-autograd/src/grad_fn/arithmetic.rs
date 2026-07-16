use std::sync::Arc;

use synaptix_core::error::Result;
use synaptix_core::grad::GradFn;
use synaptix_core::tensor::Tensor;

use crate::grad_fn::util::unbroadcast_to;

pub struct AddGradFn {
    parents: [Tensor; 2],
    shapes: [Vec<usize>; 2],
}

impl AddGradFn {
    pub fn new(lhs: &Tensor, rhs: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            shapes: [lhs.dims().to_vec(), rhs.dims().to_vec()],
            parents: [lhs.clone(), rhs.clone()],
        })
    }
}

impl GradFn for AddGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let da = unbroadcast_to(output_grad.clone(), &self.shapes[0])?;
        let db = unbroadcast_to(output_grad.clone(), &self.shapes[1])?;
        Ok(vec![Some(da), Some(db)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "AddGradFn"
    }
}

pub struct SubGradFn {
    parents: [Tensor; 2],
    shapes: [Vec<usize>; 2],
}

impl SubGradFn {
    pub fn new(lhs: &Tensor, rhs: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            shapes: [lhs.dims().to_vec(), rhs.dims().to_vec()],
            parents: [lhs.clone(), rhs.clone()],
        })
    }
}

impl GradFn for SubGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let da = unbroadcast_to(output_grad.clone(), &self.shapes[0])?;
        let neg = output_grad.neg()?;
        let db = unbroadcast_to(neg, &self.shapes[1])?;
        Ok(vec![Some(da), Some(db)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "SubGradFn"
    }
}

pub struct MulGradFn {
    parents: [Tensor; 2],
    saved: [Tensor; 2],
    shapes: [Vec<usize>; 2],
}

impl MulGradFn {
    pub fn new(lhs: &Tensor, rhs: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            shapes: [lhs.dims().to_vec(), rhs.dims().to_vec()],
            saved: [lhs.detach(), rhs.detach()],
            parents: [lhs.clone(), rhs.clone()],
        })
    }
}

impl GradFn for MulGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let da_full = output_grad.mul(&self.saved[1])?;
        let db_full = output_grad.mul(&self.saved[0])?;
        let da = unbroadcast_to(da_full, &self.shapes[0])?;
        let db = unbroadcast_to(db_full, &self.shapes[1])?;
        Ok(vec![Some(da), Some(db)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "MulGradFn"
    }
}

pub struct DivGradFn {
    parents: [Tensor; 2],
    saved: [Tensor; 2],
    shapes: [Vec<usize>; 2],
}

impl DivGradFn {
    pub fn new(lhs: &Tensor, rhs: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self {
            shapes: [lhs.dims().to_vec(), rhs.dims().to_vec()],
            saved: [lhs.detach(), rhs.detach()],
            parents: [lhs.clone(), rhs.clone()],
        })
    }
}

impl GradFn for DivGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        let da_full = output_grad.div(&self.saved[1])?;
        let neg = output_grad.neg()?;
        let a_over_b2 = self.saved[0].div(&self.saved[1])?.div(&self.saved[1])?;
        let db_full = neg.mul(&a_over_b2)?;
        let da = unbroadcast_to(da_full, &self.shapes[0])?;
        let db = unbroadcast_to(db_full, &self.shapes[1])?;
        Ok(vec![Some(da), Some(db)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "DivGradFn"
    }
}

pub struct NegGradFn {
    parents: [Tensor; 1],
}

impl NegGradFn {
    pub fn new(input: &Tensor) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()] })
    }
}

impl GradFn for NegGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(output_grad.neg()?)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "NegGradFn"
    }
}

pub struct AffineGradFn {
    parents: [Tensor; 1],
    mul: f32,
}

impl AffineGradFn {
    pub fn new(input: &Tensor, mul: f32, _add: f32) -> Arc<dyn GradFn> {
        Arc::new(Self { parents: [input.clone()], mul })
    }
}

impl GradFn for AffineGradFn {
    fn backward(&self, output_grad: &Tensor) -> Result<Vec<Option<Tensor>>> {
        Ok(vec![Some(output_grad.affine(self.mul, 0.0)?)])
    }
    fn parents(&self) -> &[Tensor] {
        &self.parents
    }
    fn name(&self) -> &'static str {
        "AffineGradFn"
    }
}

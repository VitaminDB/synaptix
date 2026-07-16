use crate::error::{Result, SynaptixError};
use crate::tensor::Tensor;

impl Tensor {
    pub fn eq(&self, _rhs: &Tensor) -> Result<Self> {
        Err(SynaptixError::Unsupported("eq: not implemented in MVP"))
    }

    pub fn gt(&self, _rhs: &Tensor) -> Result<Self> {
        Err(SynaptixError::Unsupported("gt: not implemented in MVP"))
    }

    pub fn lt(&self, _rhs: &Tensor) -> Result<Self> {
        Err(SynaptixError::Unsupported("lt: not implemented in MVP"))
    }

    pub fn allclose(&self, other: &Tensor, atol: f32, rtol: f32) -> Result<bool> {
        if self.device() != other.device() {
            return Err(SynaptixError::device_mismatch(self.device(), other.device()));
        }
        if self.dtype() != other.dtype() {
            return Err(SynaptixError::dtype_mismatch(self.dtype(), other.dtype()));
        }
        if self.dims() != other.dims() {
            return Err(SynaptixError::shape_mismatch(self.dims(), other.dims()));
        }
        if !self.device().is_cpu() {
            return Err(SynaptixError::Unsupported("allclose: cpu-only in MVP"));
        }
        let a = self.to_vec1::<f32>().or_else(|_| {
            self.flatten_all()?.to_vec1::<f32>()
        }).unwrap_or_default();
        let b = other.to_vec1::<f32>().or_else(|_| {
            other.flatten_all()?.to_vec1::<f32>()
        }).unwrap_or_default();
        if a.len() != b.len() { return Ok(false); }
        for (x, y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            let tol = atol + rtol * y.abs();
            if diff > tol { return Ok(false); }
        }
        Ok(true)
    }
}

use std::sync::RwLock;

use synaptix_core::error::{Result, SynaptixError};
use synaptix_core::tensor::Tensor;

#[derive(Debug)]
pub struct Parameter {
    tensor: RwLock<Tensor>,
    name: Option<String>,
    requires_grad: bool,
}

impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        Self { tensor: RwLock::new(tensor), name: None, requires_grad: false }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_requires_grad(mut self, requires_grad: bool) -> Self {
        self.requires_grad = requires_grad;
        self
    }

    pub fn name(&self) -> Option<&str> { self.name.as_deref() }
    pub fn requires_grad(&self) -> bool { self.requires_grad }

    pub fn tensor(&self) -> Tensor {
        let guard = self.tensor.read().expect("Parameter rwlock poisoned");
        guard.clone()
    }

    pub fn set(&self, new: Tensor) -> Result<()> {
        let mut guard = self.tensor.write().expect("Parameter rwlock poisoned");
        if guard.dims() != new.dims() {
            return Err(SynaptixError::shape_mismatch(guard.dims(), new.dims()));
        }
        if guard.dtype() != new.dtype() {
            return Err(SynaptixError::dtype_mismatch(guard.dtype(), new.dtype()));
        }
        *guard = new;
        Ok(())
    }
}

impl Clone for Parameter {
    fn clone(&self) -> Self {
        Self {
            tensor: RwLock::new(self.tensor()),
            name: self.name.clone(),
            requires_grad: self.requires_grad,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use synaptix_core::device::Device;
    use synaptix_core::dtype::DType;

    #[test]
    fn new_holds_tensor() {
        synaptix_kernels_cpu::ensure_registered();
        let t = Tensor::zeros((2usize, 3), DType::F32, Device::Cpu).unwrap();
        let p = Parameter::new(t);
        assert!(p.name().is_none());
        assert!(!p.requires_grad());
    }

    #[test]
    fn set_updates_when_compatible() {
        synaptix_kernels_cpu::ensure_registered();
        let t1 = Tensor::from_vec(vec![1.0_f32, 2.0], (2,), Device::Cpu).unwrap();
        let p = Parameter::new(t1).with_name("w");
        let t2 = Tensor::from_vec(vec![3.0_f32, 4.0], (2,), Device::Cpu).unwrap();
        p.set(t2).unwrap();
        let stored = p.tensor();
        let v = stored.to_vec1::<f32>().unwrap();
        assert_eq!(v, vec![3.0, 4.0]);
    }

    #[test]
    fn set_rejects_shape_mismatch() {
        synaptix_kernels_cpu::ensure_registered();
        let t1 = Tensor::zeros((2usize,), DType::F32, Device::Cpu).unwrap();
        let p = Parameter::new(t1);
        let t2 = Tensor::zeros((3usize,), DType::F32, Device::Cpu).unwrap();
        let res = p.set(t2);
        assert!(matches!(res, Err(SynaptixError::ShapeMismatch { .. })));
    }
}

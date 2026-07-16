use synaptix_core::tensor::Tensor;
use crate::op::OpAttrs;
use crate::error::{DispatchError, Result};

pub fn cpu_fallback(name: &str, _inputs: &[&Tensor], _attrs: &OpAttrs) -> Result<Vec<Tensor>> {
    Err(DispatchError::NoImpl(format!("no implementation for op `{name}` on this device/dtype")))
}

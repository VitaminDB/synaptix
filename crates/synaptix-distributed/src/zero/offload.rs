use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

use crate::error::Result;

/// Не реализовано (раньше молча `Ok`).
pub fn offload_param(_t: &Tensor) -> Result<()> {
    Err(crate::error::DistError::Other("offload_param: не реализовано".into()))
}

pub fn reload_param(_t: &Tensor, _device: Device) -> Result<Tensor> {
    Ok(_t.clone())
}

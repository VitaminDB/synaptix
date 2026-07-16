use synaptix_core::device::Device;
use synaptix_core::tensor::Tensor;

use crate::error::Result;

pub fn offload_param(_t: &Tensor) -> Result<()> { Ok(()) }

pub fn reload_param(_t: &Tensor, _device: Device) -> Result<Tensor> {
    Ok(_t.clone())
}

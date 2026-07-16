use synaptix_core::tensor::Tensor;
use synaptix_core::device::Device;
use crate::error::Result;

pub fn save_checkpoint(_params: &[Tensor], _path: &std::path::Path) -> Result<()> { Ok(()) }

pub fn load_checkpoint(_path: &std::path::Path, _device: Device) -> Result<Vec<Tensor>> { Ok(Vec::new()) }

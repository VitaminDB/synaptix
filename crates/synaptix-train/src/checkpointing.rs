use synaptix_core::tensor::Tensor;
use synaptix_core::device::Device;
use crate::error::Result;

use crate::error::TrainError;

// Не реализовано: раньше обе функции молча возвращали успех — «сохранённый»
// чекпойнт не писался, загрузка отдавала пустой список.
pub fn save_checkpoint(_params: &[Tensor], _path: &std::path::Path) -> Result<()> {
    Err(TrainError::Other("save_checkpoint: не реализовано".into()))
}

pub fn load_checkpoint(_path: &std::path::Path, _device: Device) -> Result<Vec<Tensor>> {
    Err(TrainError::Other("load_checkpoint: не реализовано".into()))
}

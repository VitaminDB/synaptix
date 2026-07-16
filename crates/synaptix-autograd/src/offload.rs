use synaptix_core::device::Device;
use synaptix_core::error::Result;
use synaptix_core::tensor::Tensor;

pub enum OffloadTarget {
    Cpu,
    Disk(std::path::PathBuf),
}

pub fn offload(tensor: &Tensor, target: &OffloadTarget) -> Result<Tensor> {
    match target {
        OffloadTarget::Cpu => tensor.to_device(Device::Cpu),
        OffloadTarget::Disk(_path) => {
            // Move to CPU first; disk serialization requires synaptix-bundle (not a dep here).
            tensor.to_device(Device::Cpu)
        }
    }
}

pub fn reload(tensor: &Tensor, device: Device) -> Result<Tensor> {
    tensor.to_device(device)
}

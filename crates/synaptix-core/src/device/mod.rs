use serde::{Deserialize, Serialize};

pub mod cpu;
pub mod cuda;
pub mod metal;
pub mod wgpu;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Wgpu(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceKind {
    Cpu,
    Cuda,
    Metal,
    Wgpu,
}

impl Device {
    pub fn is_cpu(self) -> bool { matches!(self, Device::Cpu) }
    pub fn is_cuda(self) -> bool { matches!(self, Device::Cuda(_)) }

    pub fn kind(self) -> DeviceKind {
        match self {
            Device::Cpu => DeviceKind::Cpu,
            Device::Cuda(_) => DeviceKind::Cuda,
            Device::Metal(_) => DeviceKind::Metal,
            Device::Wgpu(_) => DeviceKind::Wgpu,
        }
    }

    pub fn ordinal(self) -> usize {
        match self {
            Device::Cpu => 0,
            Device::Cuda(i) | Device::Metal(i) | Device::Wgpu(i) => i,
        }
    }

    pub fn same_kind(self, other: Device) -> bool { self.kind() == other.kind() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_extraction() {
        assert_eq!(Device::Cpu.kind(), DeviceKind::Cpu);
        assert_eq!(Device::Cuda(0).kind(), DeviceKind::Cuda);
        assert_eq!(Device::Cuda(3).kind(), DeviceKind::Cuda);
    }

    #[test]
    fn same_kind_check() {
        assert!(Device::Cuda(0).same_kind(Device::Cuda(7)));
        assert!(!Device::Cpu.same_kind(Device::Cuda(0)));
    }
}

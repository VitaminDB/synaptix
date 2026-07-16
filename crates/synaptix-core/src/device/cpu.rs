use crate::device::Device;

#[derive(Debug, Clone, Copy)]
pub struct CpuInfo {
    pub num_threads: usize,
}

impl CpuInfo {
    pub fn detect() -> Self {
        Self {
            num_threads: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1),
        }
    }
}

pub const fn cpu_device() -> Device { Device::Cpu }

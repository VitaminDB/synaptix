use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};

use crate::error::Result;
use super::{WeightLoader, safetensors::SafetensorsLoader};

pub struct PinnedLoader {
    inner: SafetensorsLoader,
}

impl PinnedLoader {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let inner = SafetensorsLoader::open(path)?;
        Ok(Self { inner })
    }

    pub fn open_sharded(paths: &[impl AsRef<Path>]) -> Result<Self> {
        let inner = SafetensorsLoader::open_sharded(paths)?;
        Ok(Self { inner })
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.inner = self.inner.with_device(device);
        self
    }
}

impl WeightLoader for PinnedLoader {
    fn load(&self, name: &str) -> Result<Tensor> {
        self.inner.load(name)
    }

    fn load_to(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor> {
        self.inner.load_to(name, device, dtype)
    }

    fn names(&self) -> Vec<&str> {
        self.inner.names()
    }
}

#[cfg(target_os = "linux")]
pub fn try_mlock(ptr: *const u8, len: usize) -> bool {
    extern "C" {
        fn mlock(addr: *const u8, len: usize) -> i32;
    }
    unsafe { mlock(ptr, len) == 0 }
}

#[cfg(not(target_os = "linux"))]
pub fn try_mlock(_ptr: *const u8, _len: usize) -> bool {
    false
}

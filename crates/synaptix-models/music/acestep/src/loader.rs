
use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::AceError;

pub struct CompLoader {
    inner: SynBundleLoader,
    device: Device,
}

impl CompLoader {
    pub fn open(path: impl AsRef<Path>, component: Option<&str>, device: Device) -> Result<Self, AceError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(AceError::Load(format!("not found: {}", path.display())));
        }
        let mut l = SynBundleLoader::open(path)
            .map_err(|e| AceError::Load(e.to_string()))?
            .with_device(device);
        if let Some(c) = component {
            l = l.with_component(c);
        }
        Ok(Self { inner: l, device })
    }

    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor, AceError> {
        self.inner
            .load_to(name, self.device, dtype)
            .map_err(|e| AceError::Load(format!("get '{name}': {e}")))
    }

    pub fn f32(&self, name: &str) -> Result<Tensor, AceError> {
        self.get(name, DType::F32)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}

pub fn read_bundle_file(path: impl AsRef<Path>, name: &str) -> Result<Vec<u8>, AceError> {
    let bundle = Bundle::open(path.as_ref()).map_err(|e| AceError::Load(e.to_string()))?;
    bundle
        .read_file(name)
        .map(|c| c.into_owned())
        .map_err(|e| AceError::Load(format!("read {name}: {e}")))
}

pub fn open_bundle(path: impl AsRef<Path>) -> Result<Arc<Bundle>, AceError> {
    Bundle::open(path.as_ref())
        .map(Arc::new)
        .map_err(|e| AceError::Load(e.to_string()))
}

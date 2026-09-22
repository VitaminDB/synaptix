//! Веса и файлы компонента `sheetsage2` в `.syn`-бандле.
//!
//! Компонент лежит отдельным тензорным чанком (`tensors:sheetsage2`) рядом с
//! основным — так он живёт внутри `yue2-3b.syn`, не трогая веса YuE2. Файлы
//! компонента — под `sheetsage2/` (`config.json` релиза и лицензии).

use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::{SheetError, COMPONENT};

pub struct Weights {
    inner: SynBundleLoader,
    device: Device,
}

/// Есть ли в бандле компонент SheetSage2 (не читая весов).
pub fn bundle_has_sheetsage2(path: impl AsRef<Path>) -> bool {
    Bundle::open(path.as_ref())
        .map(|b| b.tensors_slice_named(COMPONENT).is_ok())
        .unwrap_or(false)
}

impl Weights {
    pub fn open(path: impl AsRef<Path>, device: Device) -> Result<Self, SheetError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(SheetError::Load(format!("не найден: {}", path.display())));
        }
        if !bundle_has_sheetsage2(path) {
            return Err(SheetError::Load(format!(
                "в {} нет компонента SheetSage2 — допакуйте его: synaptix sheet-pack --into {}",
                path.display(),
                path.display()
            )));
        }
        let inner = SynBundleLoader::open(path)
            .map_err(|e| SheetError::Load(e.to_string()))?
            .with_device(device)
            .with_component(COMPONENT);
        Ok(Self { inner, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor, SheetError> {
        self.inner
            .load_to(name, self.device, dtype)
            .map_err(|e| SheetError::Load(format!("тензор `{name}`: {e}")))
    }

    pub fn get_on(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor, SheetError> {
        self.inner
            .load_to(name, device, dtype)
            .map_err(|e| SheetError::Load(format!("тензор `{name}`: {e}")))
    }

    pub fn f32(&self, name: &str) -> Result<Tensor, SheetError> {
        self.get(name, DType::F32)
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}

pub fn read_file(path: impl AsRef<Path>, name: &str) -> Result<Vec<u8>, SheetError> {
    let bundle = Bundle::open(path.as_ref()).map_err(|e| SheetError::Load(e.to_string()))?;
    bundle
        .read_file(name)
        .map(|c| c.into_owned())
        .map_err(|e| SheetError::Load(format!("файл `{name}`: {e}")))
}

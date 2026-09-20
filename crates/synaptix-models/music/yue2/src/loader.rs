//! Чтение весов и файлов из `.syn`-бандла.

use std::path::Path;

use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_common::{ModelError, WeightSource};

use crate::YueError;

pub struct CompLoader {
    inner: SynBundleLoader,
    device: Device,
}

impl CompLoader {
    pub fn open(path: impl AsRef<Path>, component: Option<&str>, device: Device) -> Result<Self, YueError> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(YueError::Load(format!("не найден: {}", path.display())));
        }
        let mut l = SynBundleLoader::open(path)
            .map_err(|e| YueError::Load(e.to_string()))?
            .with_device(device);
        if let Some(c) = component {
            l = l.with_component(c);
        }
        Ok(Self { inner: l, device })
    }

    pub fn get(&self, name: &str, dtype: DType) -> Result<Tensor, YueError> {
        self.get_on(name, self.device, dtype)
    }

    pub fn get_on(&self, name: &str, device: Device, dtype: DType) -> Result<Tensor, YueError> {
        self.inner
            .load_to(name, device, dtype)
            .map_err(|e| YueError::Load(format!("тензор `{name}`: {e}")))
    }

    pub fn f32(&self, name: &str) -> Result<Tensor, YueError> {
        self.get(name, DType::F32)
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}

/// `WeightSource` для `DecoderModel`: AR-ветка YuE2 лежит в бандле в обычной
/// HF-раскладке Llama (`model.layers.N.self_attn.q_proj.weight`), поэтому
/// имена не переписываются — NAR-двойники просто не запрашиваются.
pub struct BundleWeightSource {
    loader: CompLoader,
}

impl BundleWeightSource {
    pub fn new(loader: CompLoader) -> Self {
        Self { loader }
    }
}

impl WeightSource for BundleWeightSource {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        self.loader
            .get_on(key, device, dtype)
            .map_err(|e| ModelError::Load(e.to_string()))
    }
    fn contains(&self, key: &str) -> bool {
        self.loader.has(key)
    }
}

pub fn read_bundle_file(path: impl AsRef<Path>, name: &str) -> Result<Vec<u8>, YueError> {
    let bundle = Bundle::open(path.as_ref()).map_err(|e| YueError::Load(e.to_string()))?;
    bundle
        .read_file(name)
        .map(|c| c.into_owned())
        .map_err(|e| YueError::Load(format!("файл `{name}`: {e}")))
}

/// Есть ли файл в бандле (не читая его).
pub fn bundle_has_file(path: impl AsRef<Path>, name: &str) -> bool {
    Bundle::open(path.as_ref())
        .map(|b| b.read_file(name).is_ok())
        .unwrap_or(false)
}

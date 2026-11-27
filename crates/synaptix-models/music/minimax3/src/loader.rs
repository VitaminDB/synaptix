//! Веса и файлы из `.syn`-бандла по компонентам (`language_model`,
//! `rvq_depth_decoder`, `condition_encoder`, `transformer`, `vocoder`).

use std::path::Path;

use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_common::{ModelError, WeightSource};

use crate::{Music3Error, R};

pub struct Comp {
    inner: SynBundleLoader,
    device: Device,
}

impl Comp {
    pub fn open(path: &Path, component: &str, device: Device) -> R<Self> {
        let l = SynBundleLoader::open(path).map_err(|e| Music3Error::Load(format!("{}: {e}", path.display())))?.with_device(device).with_component(component);
        Ok(Self { inner: l, device })
    }

    pub fn get(&self, name: &str, dtype: DType) -> R<Tensor> {
        self.inner.load_to(name, self.device, dtype).map_err(|e| Music3Error::Load(format!("тензор `{name}`: {e}")))
    }

    pub fn has(&self, name: &str) -> bool {
        self.inner.contains(name)
    }
}

/// Источник весов для общего `DecoderModel` (HF-раскладка Qwen3).
pub struct LmSource(pub Comp);

impl WeightSource for LmSource {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        self.0.inner.load_to(key, device, dtype).map_err(|e| ModelError::Load(format!("{key}: {e}")))
    }
    fn contains(&self, key: &str) -> bool {
        self.0.has(key)
    }
    fn quant(&self, key: &str, _device: Device) -> Option<Result<synaptix_core::tensor::quant::QuantWeight, ModelError>> {
        let r = self.0.inner.prequant(key, self.0.device)?;
        Some(r.map_err(|e| ModelError::Load(e.to_string())))
    }
}

pub fn read_file(path: &Path, name: &str) -> R<Vec<u8>> {
    let b = synaptix_bundle::Bundle::open(path).map_err(|e| Music3Error::Load(e.to_string()))?;
    b.read_file(name).map(|c| c.into_owned()).map_err(|e| Music3Error::Load(format!("файл `{name}`: {e}")))
}

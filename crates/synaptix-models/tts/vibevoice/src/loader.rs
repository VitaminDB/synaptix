use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::{PreprocessorConfig, VibeVoiceConfig};
use crate::{Result, VibeVoiceError};

pub trait WeightSource {
    fn get(&self, name: &str) -> Result<Tensor>;
    fn has(&self, name: &str) -> bool;

    fn opt(&self, name: &str) -> Result<Option<Tensor>> {
        if self.has(name) {
            Ok(Some(self.get(name)?))
        } else {
            Ok(None)
        }
    }
}

pub struct LoaderSource<L: WeightLoader> {
    loader: L,
    names: HashSet<String>,
    device: Device,
    dtype: DType,
}

impl<L: WeightLoader> LoaderSource<L> {
    pub fn new(loader: L, device: Device, dtype: DType) -> Self {
        let names = loader.names().into_iter().map(|s| s.to_string()).collect();
        Self {
            loader,
            names,
            device,
            dtype,
        }
    }
}

impl<L: WeightLoader> WeightSource for LoaderSource<L> {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.loader
            .load_to(name, self.device, self.dtype)
            .map_err(|e| VibeVoiceError::Load(format!("{name}: {e}")))
    }

    fn has(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}

pub struct VibeVoiceCheckpoint {
    source: Box<dyn WeightSource>,
    pub config: VibeVoiceConfig,
    pub preprocessor: PreprocessorConfig,
    pub tokenizer_json: Vec<u8>,
    pub device: Device,
    pub dtype: DType,
}

impl VibeVoiceCheckpoint {
    pub fn open(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(VibeVoiceError::Load(format!("not found: {}", path.display())));
        }
        if path.is_dir() {
            Self::open_dir(path, device, dtype)
        } else {
            Self::open_bundle(path, device, dtype)
        }
    }

    fn open_bundle(path: &Path, device: Device, dtype: DType) -> Result<Self> {
        let bundle = Arc::new(
            Bundle::open(path).map_err(|e| VibeVoiceError::Bundle(e.to_string()))?,
        );
        let read = |name: &str| -> Result<Vec<u8>> {
            bundle
                .read_file(name)
                .map(|c| c.into_owned())
                .map_err(|e| VibeVoiceError::Bundle(format!("{name}: {e}")))
        };
        let config = VibeVoiceConfig::from_json_bytes(&read("config.json")?)?;
        let preprocessor = match read("preprocessor_config.json") {
            Ok(b) => PreprocessorConfig::from_json_bytes(&b)?,
            Err(_) => PreprocessorConfig::default(),
        };
        let tokenizer_json = read("tokenizer.json")?;
        let loader = SynBundleLoader::open(path)
            .map_err(|e| VibeVoiceError::Bundle(e.to_string()))?
            .with_component("main")
            .with_device(device);
        Ok(Self {
            source: Box::new(LoaderSource::new(loader, device, dtype)),
            config,
            preprocessor,
            tokenizer_json,
            device,
            dtype,
        })
    }

    fn open_dir(dir: &Path, device: Device, dtype: DType) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            std::fs::read(dir.join(name))
                .map_err(|e| VibeVoiceError::Load(format!("{name}: {e}")))
        };
        let config = VibeVoiceConfig::from_json_bytes(&read("config.json")?)?;
        let preprocessor = match read("preprocessor_config.json") {
            Ok(b) => PreprocessorConfig::from_json_bytes(&b)?,
            Err(_) => PreprocessorConfig::default(),
        };
        let tokenizer_json = read("tokenizer.json")?;
        let shards = scan_shards(dir).map_err(|e| VibeVoiceError::Load(e.to_string()))?;
        if shards.is_empty() {
            return Err(VibeVoiceError::Load(format!(
                "no safetensors in {}",
                dir.display()
            )));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| VibeVoiceError::Load(e.to_string()))?
            .with_device(device);
        Ok(Self {
            source: Box::new(LoaderSource::new(loader, device, dtype)),
            config,
            preprocessor,
            tokenizer_json,
            device,
            dtype,
        })
    }

    pub fn source(&self) -> &dyn WeightSource {
        self.source.as_ref()
    }
}

impl WeightSource for VibeVoiceCheckpoint {
    fn get(&self, name: &str) -> Result<Tensor> {
        self.source.get(name)
    }

    fn has(&self, name: &str) -> bool {
        self.source.has(name)
    }
}

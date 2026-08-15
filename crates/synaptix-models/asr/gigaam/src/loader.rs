//! Загрузка весов GigaAM из `.syn`-бандла или распакованного HF-снапшота.
//!
//! Гейт-путь читает распакованный каталог (`model.safetensors` + `config.json`
//! + `tokenizer.model`) через `synaptix-io` `SafetensorsLoader` (mmap). `.syn`-
//! путь читает те же артефакты компонентным чтением `Bundle`.

use std::collections::HashSet;
use std::path::Path;

use safetensors::SafeTensors;
use synaptix_bundle::Bundle;
use synaptix_core::{device::Device, dtype::DType, tensor::Tensor};
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::GigaAmConfig;
use crate::GigaAmError;

enum Source {
    Safetensors(SafetensorsLoader),
    Bundle(Box<Bundle>),
}

pub struct GigaAmWeights {
    source: Source,
    names: HashSet<String>,
    pub config: GigaAmConfig,
    pub tokenizer_model: Vec<u8>,
    pub device: Device,
    pub dtype: DType,
}

impl GigaAmWeights {
    /// Распакованный каталог HF-снапшота: `model.safetensors`, `config.json`,
    /// `tokenizer.model`.
    pub fn from_unpacked(
        dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, GigaAmError> {
        let dir = dir.as_ref();
        let weights_path = dir.join("model.safetensors");
        let config_bytes = std::fs::read(dir.join("config.json"))
            .map_err(|e| GigaAmError::Load(format!("read config.json: {e}")))?;
        let tokenizer_model = std::fs::read(dir.join("tokenizer.model"))
            .map_err(|e| GigaAmError::Load(format!("read tokenizer.model: {e}")))?;

        let config = GigaAmConfig::from_json_bytes(&config_bytes)?;

        let loader = SafetensorsLoader::open(&weights_path)
            .map_err(|e| GigaAmError::Load(format!("open {}: {e}", weights_path.display())))?
            .with_device(device);
        let names: HashSet<String> = loader.names().into_iter().map(|s| s.to_string()).collect();

        Ok(Self {
            source: Source::Safetensors(loader),
            names,
            config,
            tokenizer_model,
            device,
            dtype,
        })
    }

    /// `.syn`-бандл: веса в `tensors:main`, `config.json` и `tokenizer.model` —
    /// file-чанки.
    pub fn from_syn(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, GigaAmError> {
        let bundle = Bundle::open(path).map_err(|e| GigaAmError::Bundle(e.to_string()))?;
        let config_bytes = bundle
            .read_file("config.json")
            .map_err(|e| GigaAmError::Bundle(format!("config.json: {e}")))?
            .into_owned();
        let tokenizer_model = bundle
            .read_file("tokenizer.model")
            .map_err(|e| {
                GigaAmError::Bundle(format!(
                    "tokenizer.model: {e} — бандл `{}` не похож на модель GigaAM \
                     (выберите GigaAM `.syn`, например gigaam-v3.syn)",
                    bundle.id()
                ))
            })?
            .into_owned();
        let config = GigaAmConfig::from_json_bytes(&config_bytes)?;

        let slice = bundle
            .tensors_slice()
            .map_err(|e| GigaAmError::Bundle(e.to_string()))?;
        let st = SafeTensors::deserialize(slice)
            .map_err(|e| GigaAmError::Load(format!("safetensors header: {e}")))?;
        let names: HashSet<String> = st.names().into_iter().map(|s| s.to_string()).collect();

        Ok(Self {
            source: Source::Bundle(Box::new(bundle)),
            names,
            config,
            tokenizer_model,
            device,
            dtype,
        })
    }

    /// Тензор по имени, приведённый к compute-dtype на целевом устройстве.
    pub fn get(&self, name: &str) -> Result<Tensor, GigaAmError> {
        match &self.source {
            Source::Safetensors(loader) => loader
                .load_to(name, self.device, self.dtype)
                .map_err(|e| GigaAmError::Load(format!("'{name}': {e}"))),
            Source::Bundle(bundle) => {
                let slice = bundle
                    .tensors_slice()
                    .map_err(|e| GigaAmError::Bundle(e.to_string()))?;
                let st = SafeTensors::deserialize(slice)
                    .map_err(|e| GigaAmError::Load(format!("safetensors header: {e}")))?;
                let view = st
                    .tensor(name)
                    .map_err(|e| GigaAmError::Load(format!("st tensor '{name}': {e}")))?;
                let src_dtype = st_dtype(view.dtype()).ok_or_else(|| {
                    GigaAmError::Load(format!("st dtype {:?} unsupported", view.dtype()))
                })?;
                let t = Tensor::from_raw_slice(
                    view.data(),
                    view.shape().to_vec(),
                    src_dtype,
                    self.device,
                )
                .map_err(|e| GigaAmError::Load(format!("from_raw_slice '{name}': {e}")))?;
                if t.dtype() == self.dtype {
                    Ok(t)
                } else {
                    t.to_dtype(self.dtype)
                        .map_err(|e| GigaAmError::Load(format!("cast '{name}': {e}")))
                }
            }
        }
    }

    pub fn contains(&self, name: &str) -> bool {
        self.names.contains(name)
    }
}

fn st_dtype(d: safetensors::Dtype) -> Option<DType> {
    match d {
        safetensors::Dtype::F32 => Some(DType::F32),
        safetensors::Dtype::F16 => Some(DType::F16),
        safetensors::Dtype::BF16 => Some(DType::BF16),
        safetensors::Dtype::I32 => Some(DType::I32),
        safetensors::Dtype::I64 => Some(DType::I64),
        safetensors::Dtype::U8 => Some(DType::U8),
        safetensors::Dtype::U32 => Some(DType::U32),
        _ => None,
    }
}

pub fn enc_layer(i: usize, suffix: &str) -> String {
    format!("encoder.layers.{i}.{suffix}")
}

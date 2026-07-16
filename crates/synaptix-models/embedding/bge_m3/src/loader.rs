//! Загрузка весов BGE-M3 (XLM-RoBERTa) из распакованного HF-снапшота
//! (`model.safetensors`) через synaptix-io `SafetensorsLoader` (mmap) либо из
//! `.syn`-бандла (`tensors:main` + file-чанки config/tokenizer).

use std::collections::HashMap;
use std::path::Path;

use safetensors::SafeTensors;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_bundle::Bundle;

use crate::BgeError;

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

fn tensor_from_st(
    st: &SafeTensors,
    name: &str,
    device: Device,
    want: DType,
) -> Result<Tensor, BgeError> {
    let view = st
        .tensor(name)
        .map_err(|e| BgeError::Load(format!("st tensor '{name}': {e}")))?;
    let src_dtype = st_dtype(view.dtype())
        .ok_or_else(|| BgeError::Load(format!("st dtype {:?} unsupported", view.dtype())))?;
    let t = Tensor::from_raw_slice(view.data(), view.shape().to_vec(), src_dtype, device)
        .map_err(|e| BgeError::Load(format!("from_raw_slice '{name}': {e}")))?;
    if t.dtype() == want {
        Ok(t)
    } else {
        t.to_dtype(want)
            .map_err(|e| BgeError::Load(format!("cast '{name}': {e}")))
    }
}

/// Веса BGE-M3 (на одном устройстве, в compute-dtype).
pub struct BgeWeights {
    tensors: HashMap<String, Tensor>,
    pub device: Device,
    pub dtype: DType,
}

impl BgeWeights {
    /// Загрузить распакованный `model.safetensors` (HF-снапшот) на `device` в `dtype`.
    pub fn load_safetensors(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let path = path.as_ref();
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| BgeError::Load(format!("open {}: {e}", path.display())))?
            .with_device(device);
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let t = loader
                .load_to(&name, device, dtype)
                .map_err(|e| BgeError::Load(format!("load '{name}': {e}")))?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    /// Загрузить веса из `.syn`-бандла (`tensors:main`).
    /// Как [`Self::load_safetensors`], но снимает префикс `strip` с имён тензоров
    /// (напр. `roberta.` у `XLMRobertaForSequenceClassification` reranker'а → имена
    /// энкодера совпадают с BGE-M3-раскладкой; голова `classifier.*` остаётся как есть).
    pub fn load_safetensors_strip(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        strip: &str,
    ) -> Result<Self, BgeError> {
        let path = path.as_ref();
        let loader = SafetensorsLoader::open(path)
            .map_err(|e| BgeError::Load(format!("open {}: {e}", path.display())))?
            .with_device(device);
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let t = loader
                .load_to(&name, device, dtype)
                .map_err(|e| BgeError::Load(format!("load '{name}': {e}")))?;
            let key = name.strip_prefix(strip).unwrap_or(&name).to_string();
            tensors.insert(key, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    pub fn load_bundle(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let path = path.as_ref();
        let loader = SynBundleLoader::open(path)
            .map_err(|e| BgeError::Load(e.to_string()))?
            .with_device(device);
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let t = loader
                .load_to(&name, device, dtype)
                .map_err(|e| BgeError::Load(format!("load '{name}': {e}")))?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    /// Как [`Self::load_bundle`], но снимает префикс `strip` с имён (reranker `.syn`
    /// с `roberta.`-раскладкой → имена энкодера совпадают с BGE-M3).
    pub fn load_bundle_strip(
        path: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        strip: &str,
    ) -> Result<Self, BgeError> {
        let path = path.as_ref();
        let loader = SynBundleLoader::open(path)
            .map_err(|e| BgeError::Load(e.to_string()))?
            .with_device(device);
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let t = loader
                .load_to(&name, device, dtype)
                .map_err(|e| BgeError::Load(format!("load '{name}': {e}")))?;
            let key = name.strip_prefix(strip).unwrap_or(&name).to_string();
            tensors.insert(key, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    /// Загрузить из safetensors-байтов (`.syn` `tensors:main` слайс).
    pub fn load_safetensors_bytes(
        bytes: &[u8],
        device: Device,
        dtype: DType,
    ) -> Result<Self, BgeError> {
        let st = SafeTensors::deserialize(bytes)
            .map_err(|e| BgeError::Load(format!("deserialize st: {e}")))?;
        let names: Vec<String> = st.names().into_iter().map(|s| s.to_string()).collect();
        let mut tensors = HashMap::with_capacity(names.len());
        for name in names {
            let t = tensor_from_st(&st, &name, device, dtype)?;
            tensors.insert(name, t);
        }
        Ok(Self { tensors, device, dtype })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, BgeError> {
        self.tensors
            .get(name)
            .ok_or_else(|| BgeError::Load(format!("missing tensor '{name}'")))
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// Прочитать file-чанк из `.syn`-бандла (config.json / tokenizer.json).
pub fn read_bundle_file(bundle: &Bundle, name: &str) -> Result<Vec<u8>, BgeError> {
    bundle
        .read_file(name)
        .map(|c| c.into_owned())
        .map_err(|e| BgeError::Bundle(format!("'{name}': {e}")))
}

pub fn layer_key(i: usize, suffix: &str) -> String {
    format!("encoder.layer.{i}.{suffix}")
}

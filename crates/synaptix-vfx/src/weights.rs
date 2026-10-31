//! Веса: `.syn` (synaptix-bundle) или `model.safetensors` — тензоры f32 и метаданные (`arch`, …).

use std::collections::HashMap;
use std::path::Path;

use crate::{Error, Result};

pub struct Weights {
    pub meta: HashMap<String, String>,
    tensors: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

fn to_f32(dtype: safetensors::Dtype, data: &[u8]) -> Result<Vec<f32>> {
    use safetensors::Dtype::*;
    Ok(match dtype {
        F32 => data.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect(),
        F16 => data.chunks_exact(2).map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        BF16 => data.chunks_exact(2).map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32()).collect(),
        d => return Err(Error::Weights(format!("тип {d:?} не поддерживается"))),
    })
}

impl Weights {
    /// `.syn` — тензоры бандла; каталог — `model.safetensors` в нём; иначе — сам файл safetensors.
    pub fn open(path: &Path) -> Result<Self> {
        if path.extension().is_some_and(|e| e == "syn") {
            let b = synaptix_bundle::Bundle::open(path).map_err(|e| Error::Weights(format!("{}: {e}", path.display())))?;
            let bytes = b.tensors_slice().map_err(|e| Error::Weights(format!("{}: {e}", path.display())))?;
            return Self::parse(bytes);
        }
        let file = if path.is_dir() { path.join("model.safetensors") } else { path.to_path_buf() };
        let data = std::fs::read(&file).map_err(|e| Error::Weights(format!("{}: {e}", file.display())))?;
        Self::parse(&data)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let (_, md) = safetensors::SafeTensors::read_metadata(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let meta = md.metadata().clone().unwrap_or_default();
        let st = safetensors::SafeTensors::deserialize(bytes).map_err(|e| Error::Weights(e.to_string()))?;
        let mut tensors = HashMap::new();
        for (name, view) in st.tensors() {
            tensors.insert(name.to_string(), (view.shape().to_vec(), to_f32(view.dtype(), view.data())?));
        }
        Ok(Self { meta, tensors })
    }

    pub fn get(&self, name: &str) -> Result<&[f32]> {
        self.tensors.get(name).map(|(_, v)| v.as_slice()).ok_or_else(|| Error::Weights(format!("нет тензора {name}")))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize]> {
        self.tensors.get(name).map(|(s, _)| s.as_slice()).ok_or_else(|| Error::Weights(format!("нет тензора {name}")))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn meta_usize(&self, key: &str) -> Result<usize> {
        self.meta.get(key).and_then(|v| v.parse().ok()).ok_or_else(|| Error::Weights(format!("нет метаданных {key}")))
    }

    pub fn arch(&self) -> &str {
        self.meta.get("arch").map(String::as_str).unwrap_or("")
    }
}

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::Qwen3Config;

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

pub struct Qwen3Weights {
    pub config: Qwen3Config,
    pub tensors: HashMap<String, Tensor>,
    /// Ленивый источник — `.syn`-бандл или `.gguf`: тензоры читаются по
    /// требованию, квантованные отдаются как есть.
    pub model: Option<SynBundleLoader>,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen3Weights {
    /// Загружает все веса из директории HF (с `config.json` и
    /// `model.safetensors.index.json` либо одиночным `model.safetensors`),
    /// либо лениво открывает файл-модель (`.syn`, `.gguf`).
    pub fn load(dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        if synaptix_io::weights::is_model_file(dir) {
            let loader = SynBundleLoader::open(dir)
                .map_err(|e| LoadError::Io(e.to_string()))?
                .with_device(device);
            let cfg = loader
                .read_file("config.json")
                .ok_or_else(|| LoadError::Config("config.json: нет файла".into()))?;
            let config = Qwen3Config::from_hf_json_slice(&cfg).map_err(|e| LoadError::Config(e.to_string()))?;
            return Ok(Self { config, tensors: HashMap::new(), model: Some(loader), device, dtype });
        }
        let config = Qwen3Config::from_hf_json(dir.join("config.json"))
            .map_err(|e| LoadError::Config(e.to_string()))?;

        let shards = resolve_shards(dir)?;
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);

        let mut tensors = HashMap::with_capacity(loader.names().len());
        let names: Vec<String> = loader.names().into_iter().map(|s| s.to_string()).collect();
        for name in names {
            let t = loader
                .load_to(&name, device, dtype)
                .map_err(|e| LoadError::Io(format!("load '{name}': {e}")))?;
            tensors.insert(name, t);
        }
        Ok(Self { config, tensors, model: None, device, dtype })
    }

    pub fn get(&self, name: &str) -> Result<&Tensor, LoadError> {
        self.tensors
            .get(name)
            .ok_or_else(|| LoadError::MissingKey(name.to_string()))
    }

    pub fn layer_key(&self, layer: usize, suffix: &str) -> String {
        format!("model.layers.{layer}.{suffix}")
    }

    pub fn names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

impl synaptix_llm_common::WeightSource for Qwen3Weights {
    fn tensor(
        &self,
        key: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Tensor, synaptix_llm_common::ModelError> {
        if let Some(m) = &self.model {
            return m
                .load_to(key, device, dtype)
                .map_err(|e| synaptix_llm_common::ModelError::Load(format!("load '{key}': {e}")));
        }
        let t = self
            .get(key)
            .map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string()))?;
        if t.dtype() == dtype {
            Ok(t.clone())
        } else {
            t.to_dtype(dtype)
                .map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string()))
        }
    }

    fn contains(&self, key: &str) -> bool {
        match &self.model {
            Some(m) => m.contains(key) || m.quant_dims(key).is_some(),
            None => self.tensors.contains_key(key),
        }
    }

    fn quant(
        &self,
        key: &str,
        device: Device,
    ) -> Option<Result<synaptix_core::tensor::quant::QuantWeight, synaptix_llm_common::ModelError>> {
        let m = self.model.as_ref()?;
        m.load_quant(key, device)
            .map(|r| r.map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string())))
    }
}

fn resolve_shards(dir: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let bytes = std::fs::read(&index_path)
            .map_err(|e| LoadError::Io(format!("read index: {e}")))?;
        let idx: ShardIndex = serde_json::from_slice(&bytes)
            .map_err(|e| LoadError::Parse(format!("parse index: {e}")))?;
        let mut shards: Vec<PathBuf> = idx
            .weight_map
            .values()
            .map(|s| dir.join(s))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        if shards.is_empty() {
            return Err(LoadError::Io("empty shard index".into()));
        }
        return Ok(shards);
    }
    Err(LoadError::Io(format!(
        "no safetensors / index in {}",
        dir.display()
    )))
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("parse: {0}")]
    Parse(String),
    #[error("config: {0}")]
    Config(String),
    #[error("missing tensor: {0}")]
    MissingKey(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn qwen3_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/Qwen/Qwen3-1.7B");
        if p.join("config.json").exists() { Some(p) } else { None }
    }

    #[test]
    fn loads_qwen3_1p7b_if_present() {
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let w = Qwen3Weights::load(&dir, Device::Cpu, DType::BF16).expect("load weights");
        assert_eq!(w.config.num_hidden_layers, 28);
        assert_eq!(w.tensors.len(), 311);
        let emb = w.get("model.embed_tokens.weight").unwrap();
        assert_eq!(emb.dims(), &[w.config.vocab_size, w.config.hidden_size]);
        assert_eq!(emb.dtype(), DType::BF16);

        let q0 = w.get("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(q0.dims(), &[w.config.q_total_dim(), w.config.hidden_size]);

        let kn0 = w.get("model.layers.0.self_attn.k_norm.weight").unwrap();
        assert_eq!(kn0.dims(), &[w.config.head_dim]);
    }

    #[test]
    fn rejects_missing_dir() {
        let r = Qwen3Weights::load("/non/existent/path", Device::Cpu, DType::F32);
        assert!(r.is_err());
    }
}

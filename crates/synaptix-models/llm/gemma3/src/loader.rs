use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::SafetensorsLoader;
use synaptix_io::weights::WeightLoader;

use crate::config::Gemma3Config;

#[derive(Debug, Deserialize)]
struct ShardIndex {
    weight_map: HashMap<String, String>,
}

/// Префикс мультимодального чекпойнта Gemma-3 — текстовая башня под
/// `language_model.`. Детектится при открытии и прозрачно подставляется в [`GemmaWeights::tensor`].
const LM_PREFIX: &str = "language_model.";

/// Ленивый держатель весов: mmap-загрузчик + конфиг, БЕЗ материализации тензоров.
/// Каждый вес читается из mmap по требованию ([`tensor`](Self::tensor)) сразу на
/// целевое устройство — модель квантует на лету повесно, не держа bulk F16 в RAM.
pub struct GemmaWeights {
    pub config: Gemma3Config,
    loader: SafetensorsLoader,
    prefix: &'static str,
    pub device: Device,
    pub dtype: DType,
}

impl GemmaWeights {
    /// Открывает (mmap) шарды и парсит конфиг. Тензоры НЕ читаются. `device`/`dtype`
    /// — дефолтные подсказки (модель может грузить на другое устройство явно).
    pub fn load(dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let dir = dir.as_ref();
        let config = Gemma3Config::from_hf_json(dir.join("config.json"))
            .map_err(|e| LoadError::Config(e.to_string()))?;
        let shards = resolve_shards(dir)?;
        let loader = SafetensorsLoader::open_sharded(&shards).map_err(|e| LoadError::Io(e.to_string()))?;
        let prefix = if loader.names().iter().any(|n| n.starts_with(LM_PREFIX)) {
            LM_PREFIX
        } else {
            ""
        };
        if !loader.contains(&format!("{prefix}model.embed_tokens.weight")) {
            return Err(LoadError::MissingKey("model.embed_tokens.weight".into()));
        }
        Ok(Self { config, loader, prefix, device, dtype })
    }

    /// Лениво читает один вес из mmap на `device` в `dtype` (копируется в RAM только
    /// он). `key` — стандартное `model.*` имя; префикс `language_model.` добавляется.
    pub fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, LoadError> {
        let full = format!("{}{}", self.prefix, key);
        self.loader
            .load_to(&full, device, dtype)
            .map_err(|e| LoadError::Io(format!("load '{full}': {e}")))
    }

    pub fn contains(&self, key: &str) -> bool {
        self.loader.contains(&format!("{}{}", self.prefix, key))
    }
}

impl synaptix_llm_common::WeightSource for GemmaWeights {
    fn tensor(
        &self,
        key: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Tensor, synaptix_llm_common::ModelError> {
        GemmaWeights::tensor(self, key, device, dtype)
            .map_err(|e| synaptix_llm_common::ModelError::Load(e.to_string()))
    }

    fn contains(&self, key: &str) -> bool {
        GemmaWeights::contains(self, key)
    }
}

fn resolve_shards(dir: &Path) -> Result<Vec<PathBuf>, LoadError> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let bytes =
            std::fs::read(&index_path).map_err(|e| LoadError::Io(format!("read index: {e}")))?;
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
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    Err(LoadError::Io(format!("no safetensors / index in {}", dir.display())))
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

    fn gemma_dir() -> Option<PathBuf> {
        let p = PathBuf::from("models/gemma-3-12b-qat");
        if p.join("config.json").exists() { Some(p) } else { None }
    }

    #[test]
    fn opens_lazily_and_reads_one_tensor() {
        let Some(dir) = gemma_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        // Открытие — только mmap + парс конфига, без чтения тензоров.
        let w = GemmaWeights::load(&dir, Device::Cpu, DType::BF16).expect("open");
        assert_eq!(w.config.num_hidden_layers, 48);
        // Читаем повесно (стриминг): один тензор за раз.
        let emb = w.tensor("model.embed_tokens.weight", Device::Cpu, DType::BF16).unwrap();
        assert_eq!(emb.dims(), &[w.config.vocab_size, w.config.hidden_size]);
        let q0 = w.tensor("model.layers.0.self_attn.q_proj.weight", Device::Cpu, DType::BF16).unwrap();
        assert_eq!(q0.dims(), &[w.config.q_total_dim(), w.config.hidden_size]);
        let qn = w.tensor("model.layers.0.self_attn.q_norm.weight", Device::Cpu, DType::BF16).unwrap();
        assert_eq!(qn.dims(), &[w.config.head_dim]);
        assert!(w.contains("model.layers.0.pre_feedforward_layernorm.weight"));
        assert!(w.contains("model.layers.0.post_feedforward_layernorm.weight"));
    }
}

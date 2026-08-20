use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use synaptix_bundle::Bundle;
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
    ///
    /// `path` — HF-каталог или `.syn`-бандл (`syn-pack <hf_dir>`): у бандла
    /// шарды слиты в один tensors-чанк, а `config.json` лежит файловым чанком.
    pub fn load(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let (config, loader) = if is_bundle(path) {
            let bundle = Arc::new(
                Bundle::open(path)
                    .map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?,
            );
            let cfg_bytes = bundle
                .read_file("config.json")
                .map_err(|e| LoadError::Config(format!("{}:config.json: {e}", path.display())))?;
            let config = Gemma3Config::from_hf_json_slice(&cfg_bytes, "config.json")
                .map_err(|e| LoadError::Config(e.to_string()))?;
            let loader = SafetensorsLoader::from_bundle(bundle, None)
                .map_err(|e| LoadError::Io(e.to_string()))?;
            (config, loader)
        } else {
            let config = Gemma3Config::from_hf_json(path.join("config.json"))
                .map_err(|e| LoadError::Config(e.to_string()))?;
            let shards = resolve_shards(path)?;
            let loader = SafetensorsLoader::open_sharded(&shards)
                .map_err(|e| LoadError::Io(e.to_string()))?;
            (config, loader)
        };
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

/// `.syn`-бандл (файл) или HF-каталог — различаем по расширению.
pub fn is_bundle(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("syn"))
}

/// Прочитать вспомогательный файл модели (`tokenizer.json`, …) из каталога
/// или из `.syn`-бандла — вызывающему не нужно знать, что за раскладка.
pub fn read_aux(path: &Path, rel: &str) -> Result<Vec<u8>, LoadError> {
    if is_bundle(path) {
        let bundle =
            Bundle::open(path).map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
        return bundle
            .read_file(rel)
            .map(|c| c.into_owned())
            .map_err(|e| LoadError::Io(format!("{}:{rel}: {e}", path.display())));
    }
    let p = path.join(rel);
    std::fs::read(&p).map_err(|e| LoadError::Io(format!("read {}: {e}", p.display())))
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

    /// `.syn`-бандл Gemma (`syn-pack <hf_dir>`) отдаёт конфиг, tokenizer и
    /// веса так же, как HF-каталог: config.json/tokenizer.json лежат файловыми
    /// чанками, шарды слиты в один tensors-чанк.
    ///
    /// Запуск: `SYN_GEMMA_BUNDLE=/путь/gemma.syn cargo test -p
    /// synaptix-llm-gemma3 --lib -- --ignored bundle_load`.
    #[test]
    #[ignore = "нужен локальный .syn-бандл Gemma (SYN_GEMMA_BUNDLE)"]
    fn bundle_load_config_tokenizer_weights() {
        let Ok(path) = std::env::var("SYN_GEMMA_BUNDLE") else {
            panic!("SYN_GEMMA_BUNDLE не задан");
        };
        let path = PathBuf::from(path);
        let w = GemmaWeights::load(&path, Device::Cpu, DType::BF16).expect("открыть бандл");
        assert!(w.config.num_hidden_layers > 0);
        let tok = read_aux(&path, "tokenizer.json").expect("tokenizer.json из бандла");
        assert!(tok.len() > 1_000_000, "tokenizer.json подозрительно мал: {}", tok.len());
    }
}

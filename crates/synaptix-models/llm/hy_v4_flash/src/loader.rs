//! Веса Hy4-flash: `.syn`-бандл (`synaptix quantize` из каталога HF) или сам
//! каталог HF. Имена — как в чекпойнте tencent (`model.layers.N.…`,
//! `lm_head.weight`), источник отдаёт их как есть.

use std::path::Path;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_common::ModelError;

use crate::config::Hy4Config;

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
    #[error("weights: {0}")]
    Weights(String),
}

pub struct Hy4Weights {
    /// Сырые байты тензоров (бандл — тот же mmap): строки engram-таблиц
    /// читаются прямо из блобов, без подъёма таблицы.
    pub raw: Arc<SafetensorsLoader>,
    bundle: Option<SynBundleLoader>,
    pub config: Hy4Config,
    pub tokenizer_json: Vec<u8>,
    pub chat_template: Option<String>,
    /// Множители хэша engram по слою (`hy4_engram.json`, посчитаны numpy как в эталоне).
    pub engram_multipliers: Vec<[u64; 3]>,
}

fn werr(name: &str, e: impl std::fmt::Display) -> ModelError {
    ModelError::Load(format!("{name}: {e}"))
}

/// `hy4_engram.json`: `{"multipliers": {"2": [a, b, c], …}}` — множители хэша
/// n-грамм по номеру engram-слоя (с единицы). Эталон тянет их из
/// `numpy.random.default_rng(seed + 10007·id)`; генератор numpy тут не
/// повторяем, числа кладутся в бандл при упаковке.
fn parse_multipliers(bytes: &[u8], cfg: &Hy4Config) -> Result<Vec<[u64; 3]>, LoadError> {
    let Some(e) = &cfg.engram else { return Ok(Vec::new()) };
    let v: serde_json::Value = serde_json::from_slice(bytes).map_err(|e| LoadError::Config(format!("hy4_engram.json: {e}")))?;
    let m = v.get("multipliers").ok_or_else(|| LoadError::Config("hy4_engram.json: нет multipliers".into()))?;
    e.layer_ids
        .iter()
        .map(|id| {
            let a = m.get(id.to_string()).and_then(|x| x.as_array()).ok_or_else(|| LoadError::Config(format!("hy4_engram.json: нет слоя {id}")))?;
            let xs: Vec<u64> = a.iter().filter_map(|x| x.as_u64()).collect();
            if xs.len() != e.max_ngram {
                return Err(LoadError::Config(format!("hy4_engram.json: слой {id}: {} множителей", xs.len())));
            }
            Ok([xs[0], xs[1], xs[2]])
        })
        .collect()
}

impl Hy4Weights {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let read: Box<dyn Fn(&str) -> Option<Vec<u8>>>;
        let (raw, bundle) = if path.is_dir() {
            let shards = scan_shards(path).map_err(|e| LoadError::Io(e.to_string()))?;
            if shards.is_empty() {
                return Err(LoadError::Io(format!("нет .safetensors в {}", path.display())));
            }
            let p = path.to_path_buf();
            read = Box::new(move |n| std::fs::read(p.join(n)).ok());
            (SafetensorsLoader::open_sharded(&shards).map_err(|e| LoadError::Io(e.to_string()))?, None)
        } else {
            let b = SynBundleLoader::open(path).map_err(|e| LoadError::Io(format!("{}: {e}", path.display())))?;
            let raw = SafetensorsLoader::open_bundle(path, None).map_err(|e| LoadError::Io(e.to_string()))?;
            let rb = SynBundleLoader::open(path).map_err(|e| LoadError::Io(e.to_string()))?;
            read = Box::new(move |n| rb.read_file(n));
            (raw, Some(b))
        };
        let cfg = read("config.json").ok_or_else(|| LoadError::Io("нет config.json".into()))?;
        let config = Hy4Config::from_bytes(&cfg).map_err(LoadError::Config)?;
        let engram_multipliers = match (&config.engram, read("hy4_engram.json")) {
            (None, _) => Vec::new(),
            (Some(_), Some(b)) => parse_multipliers(&b, &config)?,
            (Some(_), None) => return Err(LoadError::Config("нет hy4_engram.json (множители хэша engram, см. docs/hy4_flash.md)".into())),
        };
        Ok(Self {
            raw: Arc::new(raw),
            bundle,
            tokenizer_json: read("tokenizer.json").unwrap_or_default(),
            chat_template: read("chat_template.jinja").and_then(|v| String::from_utf8(v).ok()),
            config,
            engram_multipliers,
        })
    }

    pub fn is_bundle(&self) -> bool {
        self.bundle.is_some()
    }

    pub fn bundle(&self) -> Option<&SynBundleLoader> {
        self.bundle.as_ref()
    }

    /// Тензор в F32 на хосте (буферы, нормы, роутер).
    pub fn host_f32(&self, name: &str) -> Result<Vec<f32>, ModelError> {
        let t = synaptix_llm_common::WeightSource::tensor(self, name, Device::Cpu, DType::F32)?;
        t.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| werr(name, e))
    }

    /// Целочисленный буфер (`lookup_table` I32) на хосте.
    pub fn host_i32(&self, name: &str) -> Result<Vec<i32>, ModelError> {
        let (bytes, dt, _) = self.raw.raw_bytes(name).ok_or_else(|| werr(name, "нет в весах"))?;
        if dt != DType::I32 {
            return Err(werr(name, format!("тип {dt:?}, ожидался I32")));
        }
        Ok(bytes.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }

    /// Стопки экспертов по слоям (зеркало RAM, ленивое чтение); плотный слой — `None`.
    pub fn expert_stacks(&self) -> Option<synaptix_llm_common::expert_mirror::ExpertStacks> {
        let b = self.bundle.as_ref()?;
        let prefixes = (0..self.config.num_hidden_layers).map(|l| (!self.config.dense[l]).then(|| format!("model.layers.{l}.mlp")));
        Some(synaptix_llm_common::expert_mirror::ExpertStacks::from_prefixes(b.clone(), prefixes))
    }
}

impl synaptix_llm_common::WeightSource for Hy4Weights {
    fn bundle_loader(&self) -> Option<SynBundleLoader> {
        self.bundle.clone()
    }

    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        let r = match &self.bundle {
            Some(b) => b.load_to(key, device, dtype),
            None => self.raw.load_to(key, device, dtype),
        };
        r.map_err(|e| werr(key, e))
    }

    fn contains(&self, key: &str) -> bool {
        match &self.bundle {
            Some(b) => b.contains(key),
            None => self.raw.tensor_info(key).is_some(),
        }
    }

    fn quant(&self, key: &str, device: Device) -> Option<Result<QuantWeight, ModelError>> {
        let r = self.bundle.as_ref()?.load_quant(key, device)?;
        Some(r.map_err(|e| werr(key, e)))
    }

    fn quant_stack(&self, key: &str, device: Device) -> Option<Result<Vec<QuantWeight>, ModelError>> {
        let r = self.bundle.as_ref()?.load_quant_stack(key, device)?;
        Some(r.map_err(|e| werr(key, e)))
    }
}

/// Открыть каталог HF шардами без бандла (проверки на исходных весах).
pub fn hf_shards(path: &Path) -> Result<Vec<std::path::PathBuf>, LoadError> {
    scan_shards(path).map_err(|e| LoadError::Io(e.to_string()))
}

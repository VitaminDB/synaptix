//! Ленивый источник весов Gemma-4: HF-каталог с шардами или `.syn`-бандл
//! (в том числе квантованный `syn-quant-v1`).
//!
//! Общий декодер спрашивает веса под именами `model.layers.N.*`, а в чекпойнте
//! Gemma-4 текстовая башня лежит под `model.language_model.*` — подмена делается
//! в [`Gemma4Weights::resolve`], чтобы остальной код о раскладке не знал.

use std::path::Path;
use std::sync::Arc;

use synaptix_bundle::Bundle;
use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_common::{ModelError, WeightSource};

use crate::config::Gemma4Config;

/// Префикс текстовой башни в мультимодальном чекпойнте.
pub const LM_PREFIX: &str = "model.language_model";
/// Префикс башни зрения.
pub const VISION_PREFIX: &str = "model.vision_tower";

enum Source {
    Files(Arc<SafetensorsLoader>),
    Bundle(SynBundleLoader),
}

pub struct Gemma4Weights {
    source: Source,
    /// В чекпойнте есть `model.language_model.*` (мультимодальная раскладка).
    /// У text-only чекпойнта веса лежат прямо под `model.*`.
    text_prefix: bool,
    pub config: Gemma4Config,
    pub tokenizer_json: Vec<u8>,
    pub chat_template: Option<String>,
    pub device: Device,
    pub dtype: DType,
}

/// `.syn`-бандл (файл) или HF-каталог — различаем по расширению.
pub fn is_bundle(path: &Path) -> bool {
    path.extension()
        .and_then(|s| s.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("syn"))
}

/// Вспомогательный файл модели из каталога или из бандла.
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

impl Gemma4Weights {
    pub fn open(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let path = path.as_ref();
        if is_bundle(path) {
            Self::open_bundle(path, device, dtype)
        } else {
            Self::open_dir(path, device, dtype)
        }
    }

    fn open_dir(path: &Path, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let config_bytes = std::fs::read(path.join("config.json"))
            .map_err(|e| LoadError::Io(format!("config.json: {e}")))?;
        let mut config = Gemma4Config::from_hf_bytes(&config_bytes)
            .map_err(|e| LoadError::Config(e.to_string()))?;
        if let Ok(gen) = std::fs::read(path.join("generation_config.json")) {
            config.merge_generation_config(&gen);
        }
        let shards = scan_shards(path).map_err(|e| LoadError::Io(e.to_string()))?;
        if shards.is_empty() {
            return Err(LoadError::Io(format!("нет .safetensors в {}", path.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);
        let source = Source::Files(Arc::new(loader));
        let text_prefix = source_contains(&source, &format!("{LM_PREFIX}.embed_tokens.weight"));
        Ok(Self {
            source,
            text_prefix,
            config,
            tokenizer_json: std::fs::read(path.join("tokenizer.json")).unwrap_or_default(),
            chat_template: std::fs::read_to_string(path.join("chat_template.jinja")).ok(),
            device,
            dtype,
        })
    }

    fn open_bundle(path: &Path, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let bundle = Bundle::open(path).map_err(|e| LoadError::Io(e.to_string()))?;
        let config_bytes = bundle
            .read_file("config.json")
            .map_err(|e| LoadError::Io(format!("config.json: {e}")))?;
        let mut config = Gemma4Config::from_hf_bytes(&config_bytes)
            .map_err(|e| LoadError::Config(e.to_string()))?;
        if let Ok(gen) = bundle.read_file("generation_config.json") {
            config.merge_generation_config(&gen);
        }
        let tokenizer_json = bundle
            .read_file("tokenizer.json")
            .map(|c| c.into_owned())
            .unwrap_or_default();
        let chat_template = bundle
            .read_file("chat_template.jinja")
            .ok()
            .and_then(|c| String::from_utf8(c.into_owned()).ok());
        drop(bundle);
        let loader = SynBundleLoader::open(path)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);
        let source = Source::Bundle(loader);
        let text_prefix = source_contains(&source, &format!("{LM_PREFIX}.embed_tokens.weight"));
        Ok(Self {
            source,
            text_prefix,
            config,
            tokenizer_json,
            chat_template,
            device,
            dtype,
        })
    }

    /// Есть ли в чекпойнте башня зрения.
    pub fn has_vision_tower(&self) -> bool {
        source_contains(&self.source, &format!("{VISION_PREFIX}.patch_embedder.input_proj.weight"))
    }

    /// `model.layers.0.…` (имена общего декодера) → фактическое имя в чекпойнте.
    fn resolve(&self, key: &str) -> String {
        if !self.text_prefix {
            return key.to_string();
        }
        match key.strip_prefix("model.") {
            Some(rest) => format!("{LM_PREFIX}.{rest}"),
            None => key.to_string(),
        }
    }
}

/// Есть ли тензор (или его квант-пара) под этим именем.
fn source_contains(source: &Source, key: &str) -> bool {
    match source {
        Source::Files(l) => l.contains(key),
        Source::Bundle(l) => {
            l.contains(key)
                || l.contains(&format!("{key}.qpacked"))
                || l.quant_dims(key).is_some()
        }
    }
}

impl WeightSource for Gemma4Weights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        let key = self.resolve(key);
        let r = match &self.source {
            Source::Files(l) => l.load_to(&key, device, dtype),
            Source::Bundle(l) => l.load_to(&key, device, dtype),
        };
        r.map_err(|e| ModelError::Load(format!("load '{key}': {e}")))
    }

    fn contains(&self, key: &str) -> bool {
        source_contains(&self.source, &self.resolve(key))
    }

    fn quant(&self, key: &str, device: Device) -> Option<Result<QuantWeight, ModelError>> {
        match &self.source {
            Source::Files(_) => None,
            Source::Bundle(l) => Some(
                l.load_quant(&self.resolve(key), device)?
                    .map_err(|e| ModelError::Load(e.to_string())),
            ),
        }
    }

    fn quant_stack(
        &self,
        key: &str,
        device: Device,
    ) -> Option<Result<Vec<QuantWeight>, ModelError>> {
        match &self.source {
            Source::Files(_) => None,
            Source::Bundle(l) => Some(
                l.load_quant_stack(&self.resolve(key), device)?
                    .map_err(|e| ModelError::Load(e.to_string())),
            ),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io: {0}")]
    Io(String),
    #[error("config: {0}")]
    Config(String),
}

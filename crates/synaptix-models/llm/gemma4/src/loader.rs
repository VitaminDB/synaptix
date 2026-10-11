//! Ленивый источник весов Gemma-4: HF-каталог с шардами или `.syn`-бандл
//! (в том числе квантованный `syn-quant-v1`).
//!
//! Общий декодер спрашивает веса под именами `model.layers.N.*`, а в чекпойнте
//! Gemma-4 текстовая башня лежит под `model.language_model.*` — подмена делается
//! в [`Gemma4Weights::resolve`], чтобы остальной код о раскладке не знал.

use std::path::Path;
use std::sync::Arc;

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::tensor::quant::QuantWeight;
use synaptix_core::tensor::Tensor;
use synaptix_io::weights::safetensors::{scan_shards, SafetensorsLoader};
use synaptix_io::weights::syn_bundle::SynBundleLoader;
use synaptix_io::weights::WeightLoader;
use synaptix_llm_common::{ModelError, RowSource, WeightSource};

use crate::config::Gemma4Config;

/// Префикс текстовой башни в мультимодальном чекпойнте.
pub const LM_PREFIX: &str = "model.language_model";
/// Префикс башни зрения.
pub const VISION_PREFIX: &str = "model.vision_tower";

enum Source {
    Files(Arc<SafetensorsLoader>),
    Bundle(Arc<SynBundleLoader>),
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
    synaptix_io::weights::is_model_file(path)
}

/// Вспомогательный файл модели из каталога, бандла или GGUF.
pub fn read_aux(path: &Path, rel: &str) -> Result<Vec<u8>, LoadError> {
    if is_bundle(path) {
        return synaptix_io::weights::read_model_file(path, rel)
            .ok_or_else(|| LoadError::Io(format!("{}:{rel}: нет файла", path.display())));
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
        let loader = SynBundleLoader::open(path)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);
        let config_bytes = loader
            .read_file("config.json")
            .ok_or_else(|| LoadError::Io("config.json: нет файла".into()))?;
        let mut config = Gemma4Config::from_hf_bytes(&config_bytes)
            .map_err(|e| LoadError::Config(e.to_string()))?;
        if let Some(gen) = loader.read_file("generation_config.json") {
            config.merge_generation_config(&gen);
        }
        let tokenizer_json = loader.read_file("tokenizer.json").unwrap_or_default();
        let chat_template = loader
            .read_file("chat_template.jinja")
            .and_then(|c| String::from_utf8(c).ok());
        let source = Source::Bundle(Arc::new(loader));
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

    /// Вес под ТОЧНЫМ именем, без подстановки префикса текстовой башни:
    /// так читаются `model.vision_tower.*` и `model.embed_vision.*`.
    pub fn raw_tensor(
        &self,
        key: &str,
        device: Device,
        dtype: DType,
    ) -> Result<Tensor, ModelError> {
        let r = match &self.source {
            Source::Files(l) => l.load_to(key, device, dtype),
            Source::Bundle(l) => l.load_to(key, device, dtype),
        };
        r.map_err(|e| ModelError::Load(format!("load '{key}': {e}")))
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

struct TableRows {
    files: Option<Arc<SafetensorsLoader>>,
    bundle: Option<Arc<SynBundleLoader>>,
    key: String,
    width: usize,
}

fn decode_row(data: &[u8], dtype: DType, w: usize, row: usize, dst: &mut [f32]) -> Result<(), ModelError> {
    let es = match dtype {
        DType::BF16 | DType::F16 => 2,
        DType::F32 => 4,
        other => return Err(ModelError::Load(format!("PLE: тип таблицы {other:?}"))),
    };
    let src = data
        .get(row * w * es..(row + 1) * w * es)
        .ok_or_else(|| ModelError::Load(format!("PLE: строка {row} за пределами таблицы")))?;
    match dtype {
        DType::BF16 => {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
            }
        }
        DType::F16 => {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(2)) {
                *d = half::f16::from_le_bytes([c[0], c[1]]).to_f32();
            }
        }
        _ => {
            for (d, c) in dst.iter_mut().zip(src.chunks_exact(4)) {
                *d = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
    }
    Ok(())
}

impl RowSource for TableRows {
    fn width(&self) -> usize {
        self.width
    }

    fn rows(&self, ids: &[u32], out: &mut [f32]) -> Result<(), ModelError> {
        let w = self.width;
        if let Some(l) = &self.files {
            let (data, dtype, _) = l.raw_bytes(&self.key).ok_or_else(|| ModelError::Load(format!("PLE: нет {}", self.key)))?;
            for (i, &id) in ids.iter().enumerate() {
                decode_row(data, dtype, w, id as usize, &mut out[i * w..(i + 1) * w])?;
            }
            return Ok(());
        }
        let l = self.bundle.as_ref().ok_or_else(|| ModelError::Load("PLE: нет источника".into()))?;
        if let Some((data, dtype, _)) = l.mapped_bytes(&self.key) {
            for (i, &id) in ids.iter().enumerate() {
                decode_row(data, dtype, w, id as usize, &mut out[i * w..(i + 1) * w])?;
            }
            return Ok(());
        }
        let (packed, scales) = l
            .quant_blob_slices(&self.key)
            .ok_or_else(|| ModelError::Load(format!("PLE: нет {}", self.key)))?
            .map_err(|e| ModelError::Load(e.to_string()))?;
        for (i, &id) in ids.iter().enumerate() {
            let r = id as usize;
            let row = packed.get(r * w..(r + 1) * w).ok_or_else(|| ModelError::Load("PLE: строка за пределами".into()))?;
            let sc = scales.get(r * 4..r * 4 + 4).ok_or_else(|| ModelError::Load("PLE: шкала за пределами".into()))?;
            let scale = f32::from_le_bytes([sc[0], sc[1], sc[2], sc[3]]);
            synaptix_core::quant::q8row::dequant_row(row, scale, &mut out[i * w..(i + 1) * w]);
        }
        Ok(())
    }
}

impl WeightSource for Gemma4Weights {
    fn rows(&self, key: &str) -> Option<Result<Arc<dyn RowSource>, ModelError>> {
        let key = self.resolve(key);
        let r = match &self.source {
            Source::Files(l) => {
                let (_, _, shape) = l.raw_bytes(&key)?;
                let width = *shape.last()?;
                l.advise_random(&key);
                TableRows { files: Some(l.clone()), bundle: None, key, width }
            }
            Source::Bundle(l) => {
                let width = match l.mapped_bytes(&key) {
                    Some((_, _, shape)) => *shape.last()?,
                    None => l.quant_dims(&key)?.2,
                };
                TableRows { files: None, bundle: Some(l.clone()), key, width }
            }
        };
        Some(Ok(Arc::new(r)))
    }

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

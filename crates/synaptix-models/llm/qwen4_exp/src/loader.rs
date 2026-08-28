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

use crate::config::Qwen4ExpConfig;
use crate::model::LM_PREFIX;
use crate::ngram::{NGramRows, TensorRows};

pub enum Source {
    Files(Arc<SafetensorsLoader>),
    Bundle(SynBundleLoader),
}

pub struct Qwen4ExpWeights {
    source: Source,
    text_prefix: bool,
    pub config: Qwen4ExpConfig,
    pub tokenizer_json: Vec<u8>,
    pub chat_template: Option<String>,
    pub device: Device,
    pub dtype: DType,
}

impl Qwen4ExpWeights {
    pub fn open(path: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let path = path.as_ref();
        if path.is_dir() {
            Self::open_dir(path, device, dtype)
        } else {
            Self::open_bundle(path, device, dtype)
        }
    }

    fn open_dir(path: &Path, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let config_bytes = std::fs::read(path.join("config.json"))
            .map_err(|e| LoadError::Io(format!("config.json: {e}")))?;
        let config = Qwen4ExpConfig::from_hf_bytes(&config_bytes)
            .map_err(|e| LoadError::Config(e.to_string()))?;
        let tokenizer_json = std::fs::read(path.join("tokenizer.json")).unwrap_or_default();
        let chat_template = std::fs::read_to_string(path.join("chat_template.jinja")).ok();
        let shards = scan_shards(path).map_err(|e| LoadError::Io(e.to_string()))?;
        if shards.is_empty() {
            return Err(LoadError::Io(format!("нет .safetensors в {}", path.display())));
        }
        let loader = SafetensorsLoader::open_sharded(&shards)
            .map_err(|e| LoadError::Io(e.to_string()))?
            .with_device(device);
        let source = Source::Files(Arc::new(loader));
        let text_prefix = !raw_contains(&source, &format!("{LM_PREFIX}.embed_tokens.weight"));
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

    fn open_bundle(path: &Path, device: Device, dtype: DType) -> Result<Self, LoadError> {
        let bundle = Bundle::open(path).map_err(|e| LoadError::Io(e.to_string()))?;
        let config_bytes = bundle
            .read_file("config.json")
            .map_err(|e| LoadError::Io(format!("config.json: {e}")))?;
        let config = Qwen4ExpConfig::from_hf_bytes(&config_bytes)
            .map_err(|e| LoadError::Config(e.to_string()))?;
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
        let text_prefix = !raw_contains(&source, &format!("{LM_PREFIX}.embed_tokens.weight"));
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

    pub fn ngram_rows(&self, layer: usize) -> Result<Box<dyn NGramRows>, ModelError> {
        let ple = self
            .config
            .ple
            .as_ref()
            .ok_or_else(|| ModelError::Load("n-gram таблица без ple-конфига".into()))?;
        let prefix = format!("{LM_PREFIX}.layers.{layer}.ple.ple_embedding.ngram_embedding");
        let single = format!("{prefix}.weight");
        if self.contains(&single) {
            let t = self.tensor(&single, Device::Cpu, DType::F32)?;
            return Ok(Box::new(TensorRows::from_tensor(&t)?));
        }
        let names: Vec<String> = (0..ple.split_parts)
            .map(|i| format!("{prefix}.shard_{i}.weight"))
            .collect();
        let names: Vec<String> = names.iter().map(|n| self.resolve(n)).collect();
        match &self.source {
            Source::Files(loader) => {
                let (_, dtype, shape) = loader
                    .raw_bytes(&names[0])
                    .ok_or_else(|| ModelError::Load(format!("нет тензора {}", names[0])))?;
                let rows = shape[0];
                let dim = shape[1];
                Ok(Box::new(MmapRows {
                    loader: loader.clone(),
                    names,
                    rows_per_shard: rows,
                    dim,
                    dtype,
                }))
            }
            Source::Bundle(_) => {
                let mut parts = Vec::with_capacity(names.len());
                for name in &names {
                    parts.push(self.tensor(name, Device::Cpu, DType::F32)?);
                }
                let refs: Vec<&Tensor> = parts.iter().collect();
                let all = Tensor::cat(&refs, 0).map_err(|e| ModelError::Load(e.to_string()))?;
                Ok(Box::new(TensorRows::from_tensor(&all)?))
            }
        }
    }
}

struct MmapRows {
    loader: Arc<SafetensorsLoader>,
    names: Vec<String>,
    rows_per_shard: usize,
    dim: usize,
    dtype: DType,
}

impl NGramRows for MmapRows {
    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> usize {
        self.rows_per_shard * self.names.len()
    }

    fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<(), ModelError> {
        let row_bytes = self.dtype.bytes_for_numel(self.dim);
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            let shard = id / self.rows_per_shard;
            let row = id % self.rows_per_shard;
            let name = self
                .names
                .get(shard)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: шард {shard} вне таблицы")))?;
            let (bytes, dtype, _) = self
                .loader
                .raw_bytes(name)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: нет тензора {name}")))?;
            let start = row * row_bytes;
            let slice = bytes
                .get(start..start + row_bytes)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: строка {id} вне шарда")))?;
            crate::ngram::decode_rows(slice, dtype, &mut out[i * self.dim..(i + 1) * self.dim]);
        }
        Ok(())
    }
}

fn raw_contains(source: &Source, key: &str) -> bool {
    match source {
        Source::Files(l) => l.contains(key),
        Source::Bundle(l) => l.names().iter().any(|n| *n == key),
    }
}

impl Qwen4ExpWeights {
    fn resolve(&self, key: &str) -> String {
        if !self.text_prefix {
            return key.to_string();
        }
        match key.strip_prefix(LM_PREFIX) {
            Some(rest) => format!("model{rest}"),
            None => key.to_string(),
        }
    }
}

impl WeightSource for Qwen4ExpWeights {
    fn tensor(&self, key: &str, device: Device, dtype: DType) -> Result<Tensor, ModelError> {
        let key = self.resolve(key);
        let r = match &self.source {
            Source::Files(l) => l.load_to(&key, device, dtype),
            Source::Bundle(l) => l.load_to(&key, device, dtype),
        };
        r.map_err(|e| ModelError::Load(format!("load '{key}': {e}")))
    }

    fn contains(&self, key: &str) -> bool {
        raw_contains(&self.source, &self.resolve(key))
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

    fn quant_stack(&self, key: &str, device: Device) -> Option<Result<Vec<QuantWeight>, ModelError>> {
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

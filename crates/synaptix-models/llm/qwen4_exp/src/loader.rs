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
use synaptix_llm_common::moe::ExpertSource;
use synaptix_llm_common::{ModelError, QLinear, WeightSource};

use crate::config::Qwen4ExpConfig;
use crate::model::LM_PREFIX;
use crate::mtp::MTP_PREFIX;
use crate::ngram::{decode_mxfp8_row, CachedRows, NGramRows, TensorRows};

pub enum Source {
    Files(Arc<SafetensorsLoader>),
    Bundle(SynBundleLoader),
}

pub struct Qwen4ExpWeights {
    source: Source,
    /// Второй взгляд на тот же mmap бандла — нужен, чтобы читать блобы
    /// `.qpacked`/`.qscales` построчно, без подъёма всего тензора.
    raw: Option<Arc<SafetensorsLoader>>,
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
            raw: None,
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
        let raw = SafetensorsLoader::open_bundle(path, None).ok().map(Arc::new);
        Ok(Self {
            source,
            raw,
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
        if let Some(table) = self.quant_ngram_rows(&names)? {
            return Ok(table);
        }
        match &self.source {
            Source::Files(loader) => {
                let (_, dtype, shape) = loader
                    .raw_bytes(&names[0])
                    .ok_or_else(|| ModelError::Load(format!("нет тензора {}", names[0])))?;
                let rows = shape[0];
                let dim = shape[1];
                for name in &names {
                    loader.advise_random(name);
                }
                let table = Box::new(MmapRows {
                    loader: loader.clone(),
                    names,
                    rows_per_shard: rows,
                    dim,
                    dtype,
                });
                Ok(match ngram_cache_bytes() {
                    0 => table,
                    bytes => Box::new(CachedRows::new(table, bytes)),
                })
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

fn ngram_cache_bytes() -> usize {
    match std::env::var("SYN_QWEN4EXP_NGRAM_CACHE_MB") {
        Ok(v) => v.trim().parse::<usize>().unwrap_or(0) * 1024 * 1024,
        Err(_) => 512 * 1024 * 1024,
    }
}

impl Qwen4ExpWeights {
    /// Один эксперт из квантованной стопки `[E, N, K]` — прямо из mmap,
    /// без подъёма всей стопки. `None` — стопка в источнике не квантована.
    pub fn quant_expert(
        &self,
        key: &str,
        expert: usize,
        device: Device,
    ) -> Option<Result<QuantWeight, ModelError>> {
        match &self.source {
            Source::Bundle(l) => Some(
                l.load_quant_expert(&self.resolve(key), expert, device)?
                    .map_err(|e| ModelError::Load(e.to_string())),
            ),
            Source::Files(_) => None,
        }
    }

    /// Готов ли источник отдавать экспертов по одному (квантованный бандл).
    pub fn has_lazy_experts(&self, layer: usize) -> bool {
        let key = format!("{LM_PREFIX}.layers.{layer}.mlp.experts.gate_up_proj");
        self.lazy_stack(&key)
    }

    /// То же про экспертов головы многотокенного предсказания.
    pub fn has_lazy_mtp_experts(&self) -> bool {
        self.lazy_stack(&format!("{MTP_PREFIX}.layers.0.mlp.experts.gate_up_proj"))
    }

    fn lazy_stack(&self, key: &str) -> bool {
        match &self.source {
            Source::Bundle(l) => l.quant_dims(&self.resolve(key)).is_some(),
            Source::Files(_) => false,
        }
    }

    /// Квантованная таблица n-грамм: строки читаются прямо из блобов
    /// `.qpacked`/`.qscales` и декодируются на лету, без подъёма шарда
    /// целиком (один шард — 380 МБ, вся таблица — полсотни гигабайт).
    fn quant_ngram_rows(&self, names: &[String]) -> Result<Option<Box<dyn NGramRows>>, ModelError> {
        let (Source::Bundle(bundle), Some(raw)) = (&self.source, &self.raw) else {
            return Ok(None);
        };
        let Some(dims) = bundle.quant_dims(&names[0]) else {
            return Ok(None);
        };
        let (slices, rows, dim) = dims;
        if slices != 1 {
            return Err(ModelError::Load(format!(
                "{}: таблица n-грамм оказалась стопкой из {slices} матриц",
                names[0]
            )));
        }
        let mut packed = Vec::with_capacity(names.len());
        let mut scales = Vec::with_capacity(names.len());
        for name in names {
            let (p, s) = resolve_quant_blobs(raw, name).ok_or_else(|| {
                ModelError::Load(format!("{name}: в бандле нет пары .qpacked/.qscales"))
            })?;
            raw.advise_random(&p);
            raw.advise_random(&s);
            packed.push(p);
            scales.push(s);
        }
        let table = Box::new(MxfpRows {
            loader: raw.clone(),
            packed,
            scales,
            rows_per_shard: rows,
            dim,
        });
        Ok(Some(match ngram_cache_bytes() {
            0 => table,
            bytes => Box::new(CachedRows::new(table, bytes)),
        }))
    }
}

fn resolve_quant_blobs(loader: &SafetensorsLoader, name: &str) -> Option<(String, String)> {
    for prefix in ["", "model."] {
        let base = format!("{prefix}{name}");
        let packed = format!("{base}.qpacked");
        let scales = format!("{base}.qscales");
        if loader.raw_bytes(&packed).is_some() && loader.raw_bytes(&scales).is_some() {
            return Some((packed, scales));
        }
    }
    None
}

/// Строки MXFP8-таблицы: `packed` — E4M3 по байту на элемент, `scales` —
/// E8M0 по байту на блок из 32.
struct MxfpRows {
    loader: Arc<SafetensorsLoader>,
    packed: Vec<String>,
    scales: Vec<String>,
    rows_per_shard: usize,
    dim: usize,
}

impl NGramRows for MxfpRows {
    fn dim(&self) -> usize {
        self.dim
    }

    fn rows(&self) -> usize {
        self.rows_per_shard * self.packed.len()
    }

    fn gather_into(&self, ids: &[i64], out: &mut [f32]) -> Result<(), ModelError> {
        let blocks = self.dim.div_ceil(32);
        for (i, id) in ids.iter().enumerate() {
            let id = *id as usize;
            let shard = id / self.rows_per_shard;
            let row = id % self.rows_per_shard;
            let packed_name = self
                .packed
                .get(shard)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: шард {shard} вне таблицы")))?;
            let scales_name = &self.scales[shard];
            let (packed, _, _) = self
                .loader
                .raw_bytes(packed_name)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: нет {packed_name}")))?;
            let (scales, _, _) = self
                .loader
                .raw_bytes(scales_name)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: нет {scales_name}")))?;
            let p = packed
                .get(row * self.dim..(row + 1) * self.dim)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: строка {id} вне шарда")))?;
            let s = scales
                .get(row * blocks..(row + 1) * blocks)
                .ok_or_else(|| ModelError::Forward(format!("n-gram: масштабы строки {id} вне шарда")))?;
            decode_mxfp8_row(p, s, &mut out[i * self.dim..(i + 1) * self.dim]);
        }
        Ok(())
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
        Source::Bundle(l) => {
            l.names().iter().any(|n| *n == key) || l.quant_dims(key).is_some()
        }
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
        // В квантованном бандле исходного тензора нет — вместо него лежит
        // пара `.qpacked`/`.qscales`, и знает о ней только манифест.
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


/// Источник экспертов поверх квантованного бандла: каждая пара
/// `gate_up`/`down` читается срезом стопки в момент промаха кэша.
pub struct BundleExperts {
    weights: Arc<Qwen4ExpWeights>,
    mtp_layer: usize,
}

impl BundleExperts {
    pub fn new(weights: Arc<Qwen4ExpWeights>) -> Self {
        let mtp_layer = weights.config.num_hidden_layers;
        Self { weights, mtp_layer }
    }

    /// Номер, под которым эксперты головы многотокенного предсказания живут в
    /// общем кэше: сразу за слоями модели.
    pub fn mtp_layer(&self) -> usize {
        self.mtp_layer
    }
}

impl ExpertSource for BundleExperts {
    fn fetch(
        &self,
        layer: usize,
        expert: usize,
        device: Device,
    ) -> Result<(QLinear, QLinear), ModelError> {
        let prefix = if layer == self.mtp_layer {
            format!("{MTP_PREFIX}.layers.0.mlp.experts")
        } else {
            format!("{LM_PREFIX}.layers.{layer}.mlp.experts")
        };
        let one = |name: &str| -> Result<QLinear, ModelError> {
            let key = format!("{prefix}.{name}");
            let w = self
                .weights
                .quant_expert(&key, expert, device)
                .ok_or_else(|| ModelError::Load(format!("{key}: стопка не квантована")))??;
            Ok(QLinear::Quant(w))
        };
        Ok((one("gate_up_proj")?, one("down_proj")?))
    }
}

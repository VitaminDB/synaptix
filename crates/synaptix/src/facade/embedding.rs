//! Embedding-фасад для KB. Нативный BGE-M3 (XLM-RoBERTa dense). `load_embedder`
//! грузит распакованный HF-снапшот и возвращает `Box<dyn Embedder>`; encode →
//! CLS-pool + L2-norm нативно (dim 1024).

use std::path::PathBuf;

use synaptix_core::dtype::DType as SynDType;
use synaptix_embedding_bge_m3::BgeM3;

pub use super::asr::ComputeDType as DType;
pub use super::asr::Device;

pub type EmbeddingResult<T> = Result<T, String>;

pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

pub trait Embedder {
    fn dim(&self) -> usize;
    fn max_tokens(&self) -> usize;
    fn encode(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>>;

    fn encode_query(&self, query: &str) -> EmbeddingResult<Vec<f32>> {
        let mut out = self.encode(&[query])?;
        Ok(out.pop().unwrap_or_default())
    }
}

#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    pub model_path: PathBuf,
    pub device: Device,
    pub dtype: DType,
    pub batch_size: usize,
}

impl EmbedderConfig {
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path, device: Device::Cpu, dtype: DType::F16, batch_size: 16 }
    }

    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
}

fn compute_to_syn(dtype: DType) -> SynDType {
    match dtype {
        DType::BF16 => SynDType::BF16,
        DType::F32 => SynDType::F32,
        DType::F16 | DType::Fp8E4M3 | DType::Nvfp4 => SynDType::F16,
    }
}

struct BgeEmbedder {
    model: BgeM3,
    batch_size: usize,
}

impl Embedder for BgeEmbedder {
    fn dim(&self) -> usize {
        self.model.dim()
    }

    fn max_tokens(&self) -> usize {
        self.model.max_tokens()
    }

    fn encode(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(texts.len());
        for chunk in texts.chunks(self.batch_size.max(1)) {
            let part = self.model.encode(chunk).map_err(|e| e.to_string())?;
            out.extend(part);
        }
        Ok(out)
    }
}

pub fn load_embedder(cfg: EmbedderConfig) -> Result<Box<dyn Embedder + Send + Sync>, String> {
    let dtype = compute_to_syn(cfg.dtype);
    let model = BgeM3::from_unpacked(&cfg.model_path, &cfg.device, dtype)
        .map_err(|e| format!("BGE-M3 load ({}): {e}", cfg.model_path.display()))?;
    Ok(Box::new(BgeEmbedder { model, batch_size: cfg.batch_size }))
}

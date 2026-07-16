//! Cross-encoder реранкер для KB — нативный `BgeReranker` (BGE-reranker-v2-m3,
//! XLM-RoBERTa + classifier).

use std::path::PathBuf;

use synaptix_core::dtype::DType as SynDType;
use synaptix_embedding_bge_m3::BgeReranker;

pub use super::asr::ComputeDType as DType;
pub use super::asr::Device;

pub type RerankResult<T> = Result<T, String>;

pub trait Reranker {
    fn max_tokens(&self) -> usize;
    fn rerank(&self, query: &str, docs: &[&str], top_k: usize) -> RerankResult<Vec<(usize, f32)>>;
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> RerankResult<Vec<f32>>;
}

#[derive(Debug, Clone)]
pub struct RerankerConfig {
    pub model_path: PathBuf,
    pub device: Device,
    pub dtype: DType,
    pub max_tokens: usize,
    pub batch_size: usize,
}

impl RerankerConfig {
    pub fn new(model_path: PathBuf) -> Self {
        Self { model_path, device: Device::Cpu, dtype: DType::F16, max_tokens: 512, batch_size: 8 }
    }
    pub fn with_device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }
    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = dtype;
        self
    }
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

fn compute_to_dtype(c: DType) -> SynDType {
    match c {
        DType::BF16 => SynDType::BF16,
        DType::F32 => SynDType::F32,
        DType::F16 | DType::Fp8E4M3 | DType::Nvfp4 => SynDType::F16,
    }
}

struct BgeRerankerAdapter {
    inner: BgeReranker,
    max_tokens: usize,
}

impl Reranker for BgeRerankerAdapter {
    fn max_tokens(&self) -> usize {
        self.max_tokens
    }
    fn rerank(&self, query: &str, docs: &[&str], top_k: usize) -> RerankResult<Vec<(usize, f32)>> {
        self.inner.rerank(query, docs, top_k).map_err(|e| e.to_string())
    }
    fn score_pairs(&self, pairs: &[(&str, &str)]) -> RerankResult<Vec<f32>> {
        self.inner.score_pairs(pairs).map_err(|e| e.to_string())
    }
}

pub fn load_reranker(cfg: RerankerConfig) -> Result<Box<dyn Reranker + Send + Sync>, String> {
    let dtype = compute_to_dtype(cfg.dtype);
    // Каталог = распакованный HF-снапшот; файл = .syn-бандл.
    let mut inner = if cfg.model_path.is_dir() {
        BgeReranker::from_unpacked(&cfg.model_path, cfg.device, dtype)
    } else {
        BgeReranker::from_syn(&cfg.model_path, cfg.device, dtype)
    }
    .map_err(|e| e.to_string())?;
    inner.set_max_tokens(cfg.max_tokens);
    Ok(Box::new(BgeRerankerAdapter { inner, max_tokens: cfg.max_tokens }))
}

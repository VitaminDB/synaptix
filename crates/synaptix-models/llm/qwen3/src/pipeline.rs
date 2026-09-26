use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_core::tensor::Tensor;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Qwen3Config;
use crate::loader::Qwen3Weights;
use crate::model::{DecoderModel, ModelError};

pub use synaptix_llm_common::generate::{GenerationConfig, GenerationStats, StreamSink};

pub struct Qwen3Pipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: Qwen3Config,
}

impl Qwen3Pipeline {
    pub fn load(model_dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        Self::load_with_max_seq(model_dir, device, dtype, None)
    }

    /// Как `load`, но RoPE-кэш строится на `max_seq` позиций (`--max-seq` для
    /// long-context). `None` → `config.max_position_embeddings`. KV-кеш BF16.
    pub fn load_with_max_seq(
        model_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        Self::load_with_opts(model_dir, device, dtype, max_seq, dtype)
    }

    /// Полный вариант: `kv_dtype` (`MXFP8` block-scale для 256K-контекста в 24GB) отдельно
    /// от compute `dtype`. `kv_dtype == dtype` → обычный BF16 KV-кеш.
    pub fn load_with_opts(
        model_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        max_seq: Option<usize>,
        kv_dtype: DType,
    ) -> Result<Self, PipelineError> {
        let mut precision = PrecisionConfig::dense(dtype);
        precision.kv = kv_dtype;
        Self::load_with_precision(model_dir, device, precision, max_seq)
    }

    /// Per-component precision (квант весов attn/mlp в NVFP4, compute=F16 и т.д.).
    /// Веса грузятся в `precision.compute`; квант-группы квантуются при загрузке.
    pub fn load_with_precision(
        model_dir: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        precision.validate().map_err(PipelineError::Load)?;
        let dir: PathBuf = model_dir.as_ref().to_path_buf();
        let weights = Qwen3Weights::load(&dir, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tok_bytes = synaptix_io::weights::read_model_file(&dir, "tokenizer.json")
            .ok_or_else(|| PipelineError::Load("tokenizer.json: нет файла".into()))?;
        let tokenizer = HfTokenizer::from_bytes(&tok_bytes)
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let dcfg = config.to_decoder_config();
        let rope_capacity = max_seq.unwrap_or(dcfg.max_position_embeddings);
        let model = DecoderModel::build_auto(
            &dcfg, &weights, device, precision.compute, precision.attn_w, precision.mlp_w, precision.lm_head, precision.embed, rope_capacity,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        .with_kv_cache_dtype(precision.kv);
        Ok(Self { model, tokenizer, config })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))?;
        Ok(enc.ids.clone())
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    fn cfg_with_eos(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        if cfg.eos_token_id.is_none() {
            cfg.eos_token_id = self.config.eos_token_id;
        }
        cfg
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate(&self.model, prompt_ids, &cfg)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_streaming(&self.model, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_streaming_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        gen_cfg: GenerationConfig,
    ) -> Result<(String, GenerationStats), PipelineError> {
        let ids = self.encode(prompt)?;
        let (new_ids, stats) = self.generate(&ids, gen_cfg)?;
        let text = self.decode(&new_ids)?;
        Ok((text, stats))
    }

    /// Годится ли модель для графового декода (см.
    /// [`synaptix_llm_common::DecoderModel::graph_decode_supported`]).
    pub fn graph_decode_supported(&self) -> bool {
        self.model.graph_decode_supported()
    }

    /// CUDA-graph decode: шаг захватывается в граф и реплеится — без
    /// launch-overhead десятков мелких ядер на токен. Работает и с MXFP8-KV.
    pub fn generate_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let mut noop = |_: u32| true;
        self.generate_with_graph_streaming(prompt_ids, gen_cfg, &mut noop)
    }

    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, prompt_ids, gen_cfg, sink)
    }

    /// Как [`Self::generate_with_graph_streaming`], но префилл стартует с
    /// `kv.seq_len` (префикс-KV) — `kv` переиспользуется между ходами чата.
    pub fn generate_with_graph_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let gen_cfg = self.cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_graph_streaming_resume(&self.model, kv, prompt_ids, &gen_cfg, sink)
            .map_err(|e| PipelineError::Forward(e.to_string()))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("load: {0}")]
    Load(String),
    #[error("model: {0}")]
    Model(String),
    #[error("tokenize: {0}")]
    Tokenize(String),
    #[error("forward: {0}")]
    Forward(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self { Self::Model(e.to_string()) }
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
    fn pipeline_loads_and_encodes() {
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load pipeline");
        let ids = p.encode("Hello").unwrap();
        assert!(!ids.is_empty());
        let back = p.decode(&ids).unwrap();
        assert!(back.contains("Hello"), "round-trip lost text: '{back}'");
    }

    #[test]
    fn pipeline_generates_one_token_greedy() {
        // Долго (~40s на prefill), пропускается если SYN_QWEN3_GENERATE не set.
        if std::env::var("SYN_QWEN3_GENERATE").is_err() {
            return;
        }
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let ids = p.encode("The capital of France is").unwrap();
        let (new_ids, stats) = p.generate(
            &ids,
            GenerationConfig { max_new_tokens: 1, temperature: 0.0, ..Default::default() },
        ).unwrap();
        assert_eq!(new_ids.len(), 1);
        let txt = p.decode(&new_ids).unwrap();
        eprintln!("[qwen3 gen] new='{txt}' prefill_ms={} decode_ms={}", stats.prefill_ms, stats.decode_ms);
    }

    #[test]
    fn chunked_prefill_matches_single_shot() {
        if std::env::var("SYN_QWEN3_GENERATE").is_err() {
            return;
        }
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let ids = p.encode("The capital of France is the city of").unwrap();
        let base = GenerationConfig { max_new_tokens: 8, temperature: 0.0, ..Default::default() };

        let (single, _) = p
            .generate(&ids, GenerationConfig { prefill_batch: 0, ..base.clone() })
            .unwrap();
        let (chunked, _) = p
            .generate(&ids, GenerationConfig { prefill_batch: 4, ..base })
            .unwrap();
        assert_eq!(single, chunked, "chunked prefill (batch=4) разошёлся с single-shot greedy");
    }

    #[test]
    fn prefix_cache_resume_matches_fresh() {
        if std::env::var("SYN_QWEN3_GENERATE").is_err() {
            return;
        }
        let Some(dir) = qwen3_dir() else { return };
        synaptix_kernels_cpu::ensure_registered();
        let p = Qwen3Pipeline::load(&dir, Device::Cpu, DType::BF16).expect("load");
        let full = p.encode("The capital of France is the city of").unwrap();
        let cap = full.len() + 16;
        let cfg = GenerationConfig {
            max_new_tokens: 8,
            temperature: 0.0,
            max_seq: Some(cap),
            ..Default::default()
        };

        let (fresh, _) = p.generate(&full, cfg.clone()).unwrap();

        let half = full.len() / 2;
        let mut kv = p.model.make_kv_cache(1, cap).unwrap();
        let t = synaptix_core::tensor::Tensor::from_vec(full[..half].to_vec(), vec![1, half], p.model.device)
            .unwrap();
        synaptix_core::grad::no_grad(|| p.model.forward(&t, &mut kv)).unwrap();
        assert_eq!(kv.seq_len, half);
        let mut noop = |_: u32| true;
        let (cached, _) = p.generate_streaming_resume(&mut kv, &full, cfg, &mut noop).unwrap();

        assert_eq!(fresh, cached, "prefix-cache resume (prefix=half) разошёлся с fresh greedy");
    }
}

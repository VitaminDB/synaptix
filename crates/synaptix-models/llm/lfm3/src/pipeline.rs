use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Lfm3Config;
use crate::loader::Lfm3Weights;
use crate::model::{DecoderModel, ModelError};

pub struct Lfm3Pipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: Lfm3Config,
}

#[derive(Debug, Clone, Copy)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
    /// Override EOS. `None` → берётся `config.eos_ids()` (Llama-3 имеет несколько).
    pub eos_token_id: Option<u32>,
    pub max_seq: Option<usize>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self { max_new_tokens: 64, temperature: 0.0, seed: 0, eos_token_id: None, max_seq: None }
    }
}

#[derive(Debug, Clone)]
pub struct GenerationStats {
    pub prompt_tokens: usize,
    pub new_tokens: usize,
    pub prefill_ms: u128,
    pub decode_ms: u128,
}

impl GenerationStats {
    fn from_common(st: synaptix_llm_common::GenerationStats) -> Self {
        Self {
            prompt_tokens: st.prompt_tokens,
            new_tokens: st.new_tokens,
            prefill_ms: st.prefill_ms,
            decode_ms: st.decode_ms,
        }
    }
}

impl Lfm3Pipeline {
    pub fn load(model_dir: impl AsRef<Path>, device: Device, dtype: DType) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций (иначе free-list
        // одного пула деградирует за длинный префилл, см.
        // `synaptix_core::device::cuda::activations_pool`).
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        Self::load_with_max_seq(model_dir, device, dtype, None)
    }

    pub fn load_with_max_seq(
        model_dir: impl AsRef<Path>,
        device: Device,
        dtype: DType,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        Self::load_with_opts(model_dir, device, dtype, max_seq, dtype)
    }

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

    /// Per-component precision (NVFP4-квант весов attn/mlp, compute=F16 и т.д.).
    /// MLX int4-веса дектвантятся в `precision.compute`; при NVFP4 затем
    /// реквантуются (см. [`crate::model::QLinear`]).
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
        let weights = Lfm3Weights::load(&dir, device)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let tok_bytes = synaptix_io::weights::read_model_file(&dir, "tokenizer.json")
            .ok_or_else(|| PipelineError::Load("tokenizer.json: нет файла".into()))?;
        let tokenizer = HfTokenizer::from_bytes(&tok_bytes)
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        let dcfg = config.to_decoder_config();
        let rope_capacity = max_seq.unwrap_or(dcfg.max_position_embeddings);
        let model = DecoderModel::build(
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

    /// Greedy/temperature generate. Возвращает только новые токены (без prompt).
    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        // Общий генератор (как у службы): на NPU — пошаговый граф декода, на CPU — префилл чанком.
        let cfg = synaptix_llm_common::GenerationConfig {
            max_new_tokens: gen_cfg.max_new_tokens,
            temperature: gen_cfg.temperature,
            seed: gen_cfg.seed,
            eos_token_id: gen_cfg.eos_token_id,
            max_seq: gen_cfg.max_seq,
            ..Default::default()
        };
        let cfg = self.common_cfg_with_eos(cfg);
        synaptix_llm_common::generate::generate(&self.model, prompt_ids, &cfg)
            .map(|(ids, st)| (ids, GenerationStats::from_common(st)))
            .map_err(|e| PipelineError::Forward(e.to_string()))
    }

    /// Стоп-токены общего декодера: если вызывающий их не задал — все EOS
    /// конфига (у Llama-3 их несколько).
    fn common_cfg_with_eos(
        &self,
        mut cfg: synaptix_llm_common::GenerationConfig,
    ) -> synaptix_llm_common::GenerationConfig {
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_ids = self.config.eos_ids();
        }
        cfg
    }

    /// Стрим token-by-token общим декодером — с полным сэмплером
    /// (top-k/top-p/min-p/штрафы), в отличие от [`Self::generate`].
    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: synaptix_llm_common::GenerationConfig,
        sink: &mut dyn synaptix_llm_common::StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        
        let cfg = self.common_cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_streaming(&self.model, prompt_ids, &cfg, sink)
            .map(|(ids, st)| (ids, GenerationStats::from_common(st)))
            .map_err(|e| PipelineError::Forward(e.to_string()))
    }

    /// Как [`Self::generate_streaming`], но префилл стартует с `kv.seq_len`
    /// (префикс-KV): история прошлого хода не считается заново. Вызывающий
    /// отвечает за то, что `prompt_ids[..kv.seq_len]` — ровно те токены, что
    /// лежат в кэше.
    pub fn generate_streaming_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: synaptix_llm_common::GenerationConfig,
        sink: &mut dyn synaptix_llm_common::StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.common_cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map(|(ids, st)| (ids, GenerationStats::from_common(st)))
            .map_err(|e| PipelineError::Forward(e.to_string()))
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

    /// CUDA-graph decode: шаг захватывается и реплеится (общий цикл
    /// `generate_graph_streaming_resume`, в т.ч. с MXFP8-KV).
    pub fn generate_with_graph(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let cfg = synaptix_llm_common::GenerationConfig {
            max_new_tokens: gen_cfg.max_new_tokens,
            temperature: gen_cfg.temperature,
            seed: gen_cfg.seed,
            eos_token_id: gen_cfg.eos_token_id,
            max_seq: gen_cfg.max_seq,
            ..Default::default()
        };
        let mut noop = |_: u32| true;
        self.generate_with_graph_streaming(prompt_ids, cfg, &mut noop)
    }

    /// Годится ли модель для графового декода.
    pub fn graph_decode_supported(&self) -> bool {
        self.model.graph_decode_supported()
    }

    /// Графовый декод со свежим кэшем.
    pub fn generate_with_graph_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: synaptix_llm_common::GenerationConfig,
        sink: &mut dyn synaptix_llm_common::StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        let kv_max = gen_cfg.max_seq.unwrap_or(prompt_ids.len() + gen_cfg.max_new_tokens);
        let mut kv = self
            .model
            .make_kv_cache(1, kv_max)
            .map_err(|e| PipelineError::Forward(e.to_string()))?;
        self.generate_with_graph_resume(&mut kv, prompt_ids, gen_cfg, sink)
    }

    /// Как [`Self::generate_streaming_resume`], но шаг декода — CUDA-граф.
    pub fn generate_with_graph_resume(
        &self,
        kv: &mut synaptix_llm_common::KvCache,
        prompt_ids: &[u32],
        gen_cfg: synaptix_llm_common::GenerationConfig,
        sink: &mut dyn synaptix_llm_common::StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("empty prompt".into()));
        }
        let cfg = self.common_cfg_with_eos(gen_cfg);
        synaptix_llm_common::generate::generate_graph_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map(|(ids, st)| (ids, GenerationStats::from_common(st)))
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


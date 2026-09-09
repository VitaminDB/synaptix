//! Пайплайн Gemma-4: токенайзер + общий декодер с расширенным профилем.

use std::path::{Path, PathBuf};

use synaptix_core::device::Device;
use synaptix_core::dtype::DType;
use synaptix_core::precision::PrecisionConfig;
use synaptix_llm_common::{
    generate, DecoderModel, GenerationConfig, GenerationStats, KvCache, ModelError, StreamSink,
};
use synaptix_tokenizer::hf::HfTokenizer;
use synaptix_tokenizer::Tokenizer;

use crate::config::Gemma4Config;
use crate::loader::{is_bundle, read_aux, Gemma4Weights};

pub struct Gemma4Pipeline {
    pub model: DecoderModel,
    pub tokenizer: HfTokenizer,
    pub config: Gemma4Config,
    /// Шаблон чата из чекпойнта (`chat_template.jinja`).
    pub chat_template: Option<String>,
}

fn load_tokenizer(path: &Path) -> Result<HfTokenizer, String> {
    if is_bundle(path) {
        let bytes = read_aux(path, "tokenizer.json").map_err(|e| e.to_string())?;
        return HfTokenizer::from_bytes(&bytes).map_err(|e| e.to_string());
    }
    HfTokenizer::from_file(path.join("tokenizer.json")).map_err(|e| e.to_string())
}

impl Gemma4Pipeline {
    pub fn load(
        model: impl AsRef<Path>,
        device: Device,
        dtype: DType,
    ) -> Result<Self, PipelineError> {
        Self::load_with_precision(model, device, PrecisionConfig::dense(dtype), None)
    }

    /// Основная точка входа: точность по компонентам, `max_seq` — ёмкость
    /// RoPE-кэша (весь `max_position_embeddings` в 256k никто не аллоцирует).
    pub fn load_with_precision(
        model: impl AsRef<Path>,
        device: Device,
        precision: PrecisionConfig,
        max_seq: Option<usize>,
    ) -> Result<Self, PipelineError> {
        // Веса — в default-пул, отдельно от пула активаций: смешанный пул за
        // длинный префилл деградирует в «решето».
        let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(device);
        let path: PathBuf = model.as_ref().to_path_buf();
        // tokenizer.json на 262k-вокаб разбирается дольше, чем грузятся веса,
        // и ни от чего не зависит — читаем его параллельно сборке модели.
        let tok_path = path.clone();
        let tok_handle = std::thread::spawn(move || load_tokenizer(&tok_path));

        // Активации Gemma переваливают за 65504 (massive activations), поэтому
        // резидуальный поток обязан быть BF16 даже при квантованных весах —
        // каст bf16↔f16 вокруг квант-ядра делает сам `QLinear::Quant`.
        let mut precision = precision;
        if precision.compute == DType::F16 {
            precision.compute = DType::BF16;
        }
        let weights = Gemma4Weights::open(&path, device, precision.compute)
            .map_err(|e| PipelineError::Load(e.to_string()))?;
        let config = weights.config.clone();
        let chat_template = weights.chat_template.clone();
        let dcfg = config.to_decoder_config();
        let rope_capacity = max_seq
            .unwrap_or(config.max_position_embeddings)
            .min(config.max_position_embeddings);
        let model = DecoderModel::build(
            &dcfg,
            &weights,
            device,
            precision.compute,
            precision.attn_w,
            precision.mlp_w,
            precision.lm_head,
            precision.embed,
            rope_capacity,
        )
        .map_err(|e| PipelineError::Model(e.to_string()))?
        .with_kv_cache_dtype(precision.kv);

        let tokenizer = tok_handle
            .join()
            .map_err(|_| PipelineError::Load("поток токенайзера упал".into()))?
            .map_err(|e| PipelineError::Load(format!("tokenizer.json: {e}")))?;
        Ok(Self { model, tokenizer, config, chat_template })
    }

    pub fn encode(&self, prompt: &str) -> Result<Vec<u32>, PipelineError> {
        self.tokenizer
            .encode(prompt, true)
            .map(|e| e.ids)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, PipelineError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| PipelineError::Tokenize(e.to_string()))
    }

    fn cfg_with_eos(&self, mut cfg: GenerationConfig) -> GenerationConfig {
        // Ход Gemma-4 закрывается не только `<eos>`: `<turn|>` (и его вариант
        // из generation_config) — такие же стоп-токены, одного первого мало.
        if cfg.eos_token_id.is_none() && cfg.eos_token_ids.is_empty() {
            cfg.eos_token_id = self.config.eos_token_ids.first().copied();
            cfg.eos_token_ids = self.config.eos_token_ids.clone();
        }
        cfg
    }

    pub fn make_kv_cache(&self, max_seq: usize) -> Result<KvCache, PipelineError> {
        self.model
            .make_kv_cache(1, max_seq)
            .map_err(|e| PipelineError::Model(e.to_string()))
    }

    pub fn generate(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        generate::generate(&self.model, prompt_ids, &cfg).map_err(PipelineError::from)
    }

    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        generate::generate_streaming(&self.model, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    /// Стрим с переиспользованием префикса: `kv.seq_len` токенов промпта уже
    /// посчитаны, префиллится только хвост.
    pub fn generate_streaming_resume(
        &self,
        kv: &mut KvCache,
        prompt_ids: &[u32],
        gen_cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(Vec<u32>, GenerationStats), PipelineError> {
        if prompt_ids.is_empty() {
            return Err(PipelineError::Tokenize("пустой промпт".into()));
        }
        let cfg = self.cfg_with_eos(gen_cfg);
        generate::generate_streaming_resume(&self.model, kv, prompt_ids, &cfg, sink)
            .map_err(PipelineError::from)
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        gen_cfg: GenerationConfig,
    ) -> Result<(String, GenerationStats), PipelineError> {
        let ids = self.encode(prompt)?;
        let (new_ids, stats) = self.generate(&ids, gen_cfg)?;
        Ok((self.decode(&new_ids)?, stats))
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
    #[error("generate: {0}")]
    Generate(String),
}

impl From<ModelError> for PipelineError {
    fn from(e: ModelError) -> Self {
        PipelineError::Generate(e.to_string())
    }
}

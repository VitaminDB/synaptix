//! Универсальный LLM-фасад: детект архитектуры → нативный pipeline (Qwen3 /
//! Qwen3-Next-Hybrid / Llama / Gemma3) за единым API. Токенайзер + jinja-шаблон
//! чата + восстановление текстовой дельты из id-токенов (DeltaSink) живут здесь,
//! чтобы и syn_chat, и CLI звали один и тот же путь. Добавление новой
//! LLM-архитектуры = новая ветка в [`LlmPipeline`]/[`load_llm`] — потребитель
//! не меняется.

use std::fmt;
use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;

use synaptix_core::dtype::DType;
use synaptix_core::precision::{parse_dtype, PrecisionConfig};
use synaptix_llm_common::{GenerationConfig, StreamSink};
use synaptix_llm_gemma3::pipeline::GemmaPipeline;
use synaptix_llm_llama::pipeline::LlamaPipeline;
use synaptix_llm_muse_glimmer::pipeline::MusePipeline;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;
use synaptix_llm_qwen3_next_hybrid::pipeline::HybridPipeline;
use synaptix_tokenizer::templates::chat_template::RenderOptions;
use synaptix_tokenizer::{
    ChatTemplate, HfTokenizer, Message as TokMessage, MessageRole, SpecialTokenKind, SpecialTokens,
    Tokenizer as _,
};

use super::arch::{config_max_seq, detect_llm_arch, read_model_file, LlmArch};

pub use synaptix_core::device::Device;
pub use synaptix_llm_common::GenerationConfig as RawGenerationConfig;

#[derive(Debug, Clone)]
pub struct LlmError(pub String);

impl fmt::Display for LlmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for LlmError {}

impl From<&str> for LlmError {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for LlmError {
    fn from(s: String) -> Self {
        Self(s)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvDtypePolicy {
    F16,
    BF16,
    F32,
}

impl KvDtypePolicy {
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" | "fp32" => Some(Self::F32),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
        }
    }

    fn to_dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F32 => DType::F32,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TiedEmbeddingsMode {
    Auto,
    Tied,
    Untied,
}

impl TiedEmbeddingsMode {
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "tied" => Some(Self::Tied),
            "untied" => Some(Self::Untied),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Tied => "tied",
            Self::Untied => "untied",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FlashAttnMode {
    Off,
    #[default]
    Fa4,
}

impl FromStr for FlashAttnMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "fa4" => Ok(Self::Fa4),
            other => Err(format!("неизвестный FlashAttnMode: {other}")),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerSyncMode {
    #[default]
    Auto,
    Off,
    On,
}

impl FromStr for LayerSyncMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            "on" => Ok(Self::On),
            other => Err(format!("неизвестный LayerSyncMode: {other}")),
        }
    }
}

#[derive(Debug, Clone)]
pub struct QuantPolicy {
    pub weights_storage: DType,
    pub compute: DType,
    pub kv_dtype: KvDtypePolicy,
    pub lm_head_storage: DType,
    pub embed_storage: DType,
    pub tied_embeddings: TiedEmbeddingsMode,
    pub ssm_state_dtype: KvDtypePolicy,
    pub conv_state_dtype: KvDtypePolicy,
    pub preset_name: String,
}

impl QuantPolicy {
    pub fn quality() -> Self {
        Self {
            weights_storage: DType::BF16,
            compute: DType::BF16,
            kv_dtype: KvDtypePolicy::BF16,
            lm_head_storage: DType::BF16,
            embed_storage: DType::BF16,
            tied_embeddings: TiedEmbeddingsMode::Auto,
            ssm_state_dtype: KvDtypePolicy::F32,
            conv_state_dtype: KvDtypePolicy::F16,
            preset_name: "quality".to_string(),
        }
    }

    pub fn balance() -> Self {
        Self {
            weights_storage: DType::NVFP4,
            compute: DType::F16,
            kv_dtype: KvDtypePolicy::F16,
            lm_head_storage: DType::MXFP8,
            embed_storage: DType::MXFP8,
            tied_embeddings: TiedEmbeddingsMode::Auto,
            ssm_state_dtype: KvDtypePolicy::F32,
            conv_state_dtype: KvDtypePolicy::F16,
            preset_name: "balance".to_string(),
        }
    }

    pub fn vram_saver() -> Self {
        Self {
            weights_storage: DType::NVFP4,
            compute: DType::F16,
            kv_dtype: KvDtypePolicy::F16,
            lm_head_storage: DType::NVFP4,
            embed_storage: DType::NVFP4,
            tied_embeddings: TiedEmbeddingsMode::Auto,
            ssm_state_dtype: KvDtypePolicy::F32,
            conv_state_dtype: KvDtypePolicy::F16,
            preset_name: "vram_saver".to_string(),
        }
    }

    pub fn detect_preset(&self) -> String {
        if self.preset_name.is_empty() {
            "custom".to_string()
        } else {
            self.preset_name.clone()
        }
    }

    pub fn to_precision(&self) -> Result<PrecisionConfig, LlmError> {
        let mut p = match self.weights_storage {
            DType::NVFP4 => PrecisionConfig::nvfp4(),
            DType::MXFP8 => PrecisionConfig::mxfp8(),
            _ => PrecisionConfig::dense(self.compute),
        };
        p.kv = self.kv_dtype.to_dtype();
        p.lm_head = self.lm_head_storage;
        p.embed = self.embed_storage;
        p.validate().map_err(LlmError::from)?;
        Ok(p)
    }
}

/// `--kv-dtype` → DType KV-кеша. `fp8`/`mxfp8` → MXFP8 (Blackwell block-scale);
/// иначе compute dtype.
pub fn parse_kv_dtype(s: Option<&str>, compute: DType) -> DType {
    match s.map(|x| x.to_ascii_lowercase()).as_deref() {
        Some("fp8") | Some("mxfp8") => DType::MXFP8,
        _ => compute,
    }
}

/// Строит [`PrecisionConfig`] из CLI-стиля: пресет (`quant`) → override compute →
/// override весов (storage/lm-head/embed) → kv. Override приоритетнее пресета.
/// Дефолт = dense BF16 (none-пресет без compute_dtype). Общий для CLI и synthos.
pub fn build_precision(
    quant: Option<&str>,
    compute_dtype: Option<&str>,
    storage_dtype: Option<&str>,
    lm_head_dtype: Option<&str>,
    embed_dtype: Option<&str>,
    kv_dtype: Option<&str>,
) -> Result<PrecisionConfig, String> {
    let quant = quant.unwrap_or("none").to_ascii_lowercase();
    let preset = PrecisionConfig::from_preset(&quant)
        .ok_or_else(|| format!("unknown quant '{quant}' (none|nvfp4|fp8|mxfp8)"))?;

    let compute = match compute_dtype {
        Some(s) => parse_dtype(s).ok_or_else(|| format!("bad compute-dtype '{s}'"))?,
        None if quant == "none" => DType::BF16,
        None => preset.compute,
    };

    let mut p = if quant == "none" {
        PrecisionConfig::dense(compute)
    } else {
        PrecisionConfig { compute, ..preset }
    };

    if let Some(s) = storage_dtype {
        let dt = parse_dtype(s).ok_or_else(|| format!("bad storage-dtype '{s}'"))?;
        p.attn_w = dt;
        p.mlp_w = dt;
    }
    if let Some(s) = lm_head_dtype {
        p.lm_head = parse_dtype(s).ok_or_else(|| format!("bad lm-head-dtype '{s}'"))?;
    }
    if let Some(s) = embed_dtype {
        p.embed = parse_dtype(s).ok_or_else(|| format!("bad embed-dtype '{s}'"))?;
    }
    p.kv = parse_kv_dtype(kv_dtype, p.compute);
    p.validate()?;
    Ok(p)
}

#[derive(Debug, Clone)]
pub struct GenerationOptions {
    pub max_new_tokens: usize,
    pub max_seq_len: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub seed: u64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into() }
    }
    pub fn tool(content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into() }
    }

    fn to_tok(&self) -> TokMessage {
        match self.role.as_str() {
            "system" => TokMessage::system(self.content.clone()),
            "assistant" => TokMessage::assistant(self.content.clone()),
            "tool" => TokMessage {
                role: MessageRole::Tool,
                content: self.content.clone(),
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            _ => TokMessage::user(self.content.clone()),
        }
    }
}

pub struct LlmConfig {
    pub max_seq_len: usize,
}

// ── Нативный pipeline (Qwen3 / Hybrid / Llama / Gemma3) ──────────────────────

enum LlmPipeline {
    Qwen3(Qwen3Pipeline),
    Hybrid(HybridPipeline),
    Llama(LlamaPipeline),
    Gemma3(GemmaPipeline),
    MuseGlimmer(MusePipeline),
}

impl LlmPipeline {
    fn rope_capacity(&self) -> usize {
        match self {
            Self::Qwen3(p) => p.model.rope_capacity(),
            Self::Hybrid(p) => p.model.rope_capacity(),
            Self::Llama(p) => p.model.rope_capacity(),
            Self::Gemma3(p) => p.model.rope_capacity(),
            Self::MuseGlimmer(p) => p.model.rope_capacity(),
        }
    }

    fn kv_bytes_per_token(&self) -> usize {
        match self {
            Self::Qwen3(p) => p.model.kv_bytes_per_token(),
            Self::Hybrid(p) => p.model.kv_bytes_per_token(),
            Self::Llama(p) => p.model.kv_bytes_per_token(),
            Self::Gemma3(p) => p.model.kv_bytes_per_token(),
            Self::MuseGlimmer(p) => p.model.kv_bytes_per_token(),
        }
    }

    fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(), LlmError> {
        match self {
            LlmPipeline::Qwen3(p) => p
                .generate_streaming(prompt_ids, cfg, sink)
                .map(|_| ())
                .map_err(|e| LlmError(e.to_string())),
            LlmPipeline::MuseGlimmer(p) => p
                .generate_streaming(prompt_ids, cfg, sink)
                .map(|_| ())
                .map_err(|e| LlmError(e.to_string())),
            LlmPipeline::Hybrid(p) => {
                if mtp_enabled() && p.has_mtp() {
                    {
                        return p
                            .generate_mtp_with_graph(prompt_ids, cfg, sink)
                            .map(|_| ())
                            .map_err(|e| LlmError(e.to_string()));
                    }
                }
                if graph_decode_enabled() && p.graph_decode_supported() {
                    return p
                        .generate_with_graph_streaming(prompt_ids, cfg, sink)
                        .map(|_| ())
                        .map_err(|e| LlmError(e.to_string()));
                }
                p.generate_streaming(prompt_ids, cfg, sink)
                    .map(|_| ())
                    .map_err(|e| LlmError(e.to_string()))
            }
            // Llama/Gemma3 ещё не имеют нативного token-by-token стрима: гоняем
            // eager generate и прокручиваем полученные id через sink (псевдо-стрим;
            // сэмплинг top_k/top_p/min_p/repeat_penalty этими пайплайнами пока не
            // поддержан и опускается — для syn_chat сегодня это Qwen3-only путь).
            LlmPipeline::Llama(p) => {
                use synaptix_llm_llama::pipeline::GenerationConfig as LCfg;
                let lcfg = LCfg {
                    max_new_tokens: cfg.max_new_tokens,
                    temperature: cfg.temperature,
                    seed: cfg.seed,
                    eos_token_id: cfg.eos_token_ids.first().copied(),
                    max_seq: cfg.max_seq,
                };
                let (new_ids, _) =
                    p.generate(prompt_ids, lcfg).map_err(|e| LlmError(e.to_string()))?;
                for id in new_ids {
                    if !sink.on_token(id) {
                        break;
                    }
                }
                Ok(())
            }
            LlmPipeline::Gemma3(p) => {
                use synaptix_llm_gemma3::pipeline::GenerationConfig as GCfg;
                let gcfg = GCfg {
                    max_new_tokens: cfg.max_new_tokens,
                    temperature: cfg.temperature,
                    seed: cfg.seed,
                    eos_token_id: cfg.eos_token_ids.first().copied(),
                    max_seq: cfg.max_seq,
                };
                let (new_ids, _) =
                    p.generate(prompt_ids, gcfg).map_err(|e| LlmError(e.to_string()))?;
                for id in new_ids {
                    if !sink.on_token(id) {
                        break;
                    }
                }
                Ok(())
            }
        }
    }
}

pub struct Llm {
    pipeline: Mutex<LlmPipeline>,
    config: LlmConfig,
    device: Device,
    vocab_size: usize,
}

impl Llm {
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn config(&self) -> &LlmConfig {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        self.pipeline
            .lock()
            .map(|p| p.kv_bytes_per_token())
            .unwrap_or(0)
    }

    /// Сырой стриминг: нативный pipeline шлёт id токенов в `sink`; декод/UI —
    /// на стороне вызывающего. Для callback-API (текстовая дельта) —
    /// [`LlmGeneration::generate_streaming`].
    pub fn generate_streaming(
        &self,
        prompt_ids: &[u32],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(), LlmError> {
        let pipeline = self
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        pipeline.generate_streaming(prompt_ids, cfg, sink)
    }
}

pub struct LlmTokenizer {
    tokenizer: HfTokenizer,
    template: Option<ChatTemplate>,
    eos_ids: Vec<u32>,
}

impl LlmTokenizer {
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, LlmError> {
        self.tokenizer
            .encode(text, false)
            .map(|e| e.ids.clone())
            .map_err(|e| LlmError(e.to_string()))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String, LlmError> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| LlmError(e.to_string()))
    }

    pub fn apply_chat_template_ex_tools(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        enable_thinking: bool,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String, LlmError> {
        let msgs: Vec<TokMessage> = messages.iter().map(Message::to_tok).collect();
        match &self.template {
            Some(tmpl) => {
                let mut opts = RenderOptions::new()
                    .with_generation_prompt(add_generation_prompt)
                    .with_var("enable_thinking", serde_json::Value::Bool(enable_thinking));
                if let Some(t) = tools {
                    if !t.is_empty() {
                        opts = opts.with_var("tools", serde_json::Value::Array(t.to_vec()));
                    }
                }
                tmpl.render(&msgs, &opts).map_err(|e| LlmError(e.to_string()))
            }
            None => Ok(fallback_render(messages, add_generation_prompt)),
        }
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }
}

/// ChatML-fallback, если у модели нет jinja-шаблона.
fn fallback_render(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut s = String::new();
    for m in messages {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content));
    }
    if add_generation_prompt {
        s.push_str("<|im_start|>assistant\n");
    }
    s
}

pub struct LlmGeneration<'a> {
    model: &'a Llm,
    opts: GenerationOptions,
    stop_sequences: Vec<String>,
    stop_tokens: Vec<u32>,
}

impl<'a> LlmGeneration<'a> {
    pub fn new(model: &'a Llm, opts: GenerationOptions) -> Self {
        Self { model, opts, stop_sequences: Vec::new(), stop_tokens: Vec::new() }
    }

    pub fn add_stop_sequence(&mut self, seq: &str) {
        if !seq.is_empty() {
            self.stop_sequences.push(seq.to_string());
        }
    }

    pub fn set_stop_tokens(&mut self, tokens: Vec<u32>) {
        self.stop_tokens = tokens;
    }

    pub fn generate_streaming<F>(
        &mut self,
        prompt_ids: &[u32],
        tokenizer: &LlmTokenizer,
        on_token: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let cfg = GenerationConfig {
            max_new_tokens: self.opts.max_new_tokens,
            temperature: self.opts.temperature,
            top_k: self.opts.top_k,
            top_p: self.opts.top_p,
            min_p: self.opts.min_p,
            repetition_penalty: self.opts.repeat_penalty,
            seed: self.opts.seed,
            eos_token_id: None,
            eos_token_ids: self.stop_tokens.clone(),
            max_seq: Some(self.opts.max_seq_len.max(1)),
            prefill_batch: prefill_chunk_size(),
        };

        let mut sink = DeltaSink {
            tokenizer,
            on_token,
            acc: Vec::new(),
            decoded: String::new(),
            stop_sequences: &self.stop_sequences,
        };

        let pipeline = self
            .model
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        pipeline.generate_streaming(prompt_ids, cfg, &mut sink)
    }
}

/// Sink, восстанавливающий текстовую дельту из id-токенов нативного стрима.
/// На каждый токен: накапливает id, декодит весь буфер, диффит против прежнего
/// decoded → дельта. Зовёт `on_token(id, delta)`; `false` → отмена. Накопленный
/// текст сверяется со стоп-строками (`</tool_call>` и т.п.) — совпадение → стоп.
struct DeltaSink<'t, 's, F: FnMut(u32, &str) -> bool> {
    tokenizer: &'t LlmTokenizer,
    on_token: F,
    acc: Vec<u32>,
    decoded: String,
    stop_sequences: &'s [String],
}

impl<'t, 's, F: FnMut(u32, &str) -> bool> StreamSink for DeltaSink<'t, 's, F> {
    fn on_token(&mut self, token_id: u32) -> bool {
        self.acc.push(token_id);
        let full = match self.tokenizer.decode(&self.acc) {
            Ok(s) => s,
            Err(_) => return true,
        };
        let delta: String =
            if full.len() >= self.decoded.len() && full.starts_with(&self.decoded) {
                full[self.decoded.len()..].to_string()
            } else {
                String::new()
            };
        self.decoded = full;

        if !(self.on_token)(token_id, &delta) {
            return false;
        }
        for stop in self.stop_sequences {
            if self.decoded.ends_with(stop.as_str()) {
                return false;
            }
        }
        true
    }
}

// ── Загрузка ────────────────────────────────────────────────────────────────

/// Источник jinja-шаблона: `chat_template.jinja` либо
/// `tokenizer_config.json["chat_template"]`. None → fallback ChatML.
fn load_template_source(model: &Path) -> Option<String> {
    if let Some(bytes) = read_model_file(model, "chat_template.jinja") {
        if let Ok(s) = String::from_utf8(bytes) {
            if !s.trim().is_empty() {
                return Some(s);
            }
        }
    }
    let cfg = read_model_file(model, "tokenizer_config.json")?;
    let v: serde_json::Value = serde_json::from_slice(&cfg).ok()?;
    v.get("chat_template").and_then(|t| t.as_str()).map(str::to_string)
}

fn build_tokenizer(model: &Path) -> Result<HfTokenizer, LlmError> {
    let bytes = read_model_file(model, "tokenizer.json")
        .ok_or_else(|| LlmError(format!("tokenizer.json не найден в {}", model.display())))?;
    HfTokenizer::from_bytes(&bytes).map_err(|e| LlmError(format!("tokenizer.json: {e}")))
}

/// Стоп-токены: eos из specials + `<|im_end|>` (конец хода чата).
fn build_eos_ids(tok: &HfTokenizer, specials: &SpecialTokens) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let mut push = |id: Option<u32>| {
        if let Some(i) = id {
            if !ids.contains(&i) {
                ids.push(i);
            }
        }
    };
    push(specials.eos_id());
    push(specials.id_of(SpecialTokenKind::ImEnd));
    push(tok.token_to_id("<|im_end|>"));
    push(tok.token_to_id("<|eot|>"));
    ids
}

fn ensure_kernels_registered() {
    use std::sync::OnceLock;
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        synaptix_kernels_cpu::ensure_registered();
        synaptix_kernels_cuda::ensure_registered();
    });
}

fn build_facade(
    pipeline: LlmPipeline,
    path: &Path,
    device: Device,
    max_seq: Option<usize>,
) -> Result<(Llm, LlmTokenizer), LlmError> {
    let tokenizer = build_tokenizer(path)?;
    let specials = tokenizer.special_tokens().clone();
    let eos_ids = build_eos_ids(&tokenizer, &specials);
    let vocab_size = tokenizer.vocab_size(true);
    let template = load_template_source(path)
        .map(|src| ChatTemplate::from_source_with_specials(src, specials.clone()));

    let capacity = pipeline.rope_capacity();
    let max_seq_len = max_seq
        .or_else(|| config_max_seq(path))
        .map_or(capacity, |requested| requested.min(capacity));

    let model = Llm {
        pipeline: Mutex::new(pipeline),
        config: LlmConfig { max_seq_len },
        device,
        vocab_size,
    };
    let tok = LlmTokenizer { tokenizer, template, eos_ids };
    Ok((model, tok))
}

pub fn load_llm(
    path: &Path,
    device: Device,
    precision: PrecisionConfig,
    max_seq: Option<usize>,
) -> Result<(Llm, LlmTokenizer), LlmError> {
    ensure_kernels_registered();

    let arch = detect_llm_arch(path).map_err(LlmError)?;
    let pipeline = match arch {
        LlmArch::Qwen3 => Qwen3Pipeline::load_with_precision(path, device, precision, max_seq)
            .map(LlmPipeline::Qwen3)
            .map_err(|e| LlmError(format!("load qwen3: {e}")))?,
        LlmArch::Hybrid => HybridPipeline::load_with_precision_mtp(
            path,
            device,
            precision,
            max_seq,
            mtp_enabled(),
        )
        .map(LlmPipeline::Hybrid)
        .map_err(|e| LlmError(format!("load hybrid: {e}")))?,
        LlmArch::Llama => LlamaPipeline::load_with_precision(path, device, precision, max_seq)
            .map(LlmPipeline::Llama)
            .map_err(|e| LlmError(format!("load llama: {e}")))?,
        LlmArch::Gemma3 => GemmaPipeline::load_with_precision(path, device, precision, max_seq)
            .map(LlmPipeline::Gemma3)
            .map_err(|e| LlmError(format!("load gemma3: {e}")))?,
        LlmArch::MuseGlimmer => {
            MusePipeline::load_with_precision(path, device, precision, max_seq)
                .map(LlmPipeline::MuseGlimmer)
                .map_err(|e| LlmError(format!("load muse_glimmer: {e}")))?
        }
    };

    build_facade(pipeline, path, device, max_seq)
}

/// Удобная загрузка из квант-политики (synthos-путь): строит precision из
/// [`QuantPolicy`], берёт `max_seq` из config.json.
pub fn load_llm_with_policy(
    path: &Path,
    policy: QuantPolicy,
    device: &Device,
) -> Result<(Llm, LlmTokenizer), LlmError> {
    let precision = policy.to_precision()?;
    let max_seq = config_max_seq(path);
    load_llm(path, *device, precision, max_seq)
}

// ── Рантайм-сеттеры (no-op: нативный synaptix управляет иначе) ───────────────

static MTP_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_mtp_enabled(on: bool) {
    MTP_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn mtp_enabled() -> bool {
    MTP_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

pub fn set_flash_attn_mode(_mode: FlashAttnMode) {}
static GRAPH_DECODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn set_graph_decode_enabled(on: bool) {
    GRAPH_DECODE.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn graph_decode_enabled() -> bool {
    GRAPH_DECODE.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_la_prep_fused_disabled(_off: bool) {}
pub fn set_gdr_fused_disabled(_off: bool) {}
static PREFILL_CHUNK: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub fn set_prefill_chunk_size(size: usize) {
    PREFILL_CHUNK.store(size, std::sync::atomic::Ordering::Relaxed);
}

pub fn prefill_chunk_size() -> usize {
    PREFILL_CHUNK.load(std::sync::atomic::Ordering::Relaxed)
}
pub fn set_layer_sync_mode(mode: LayerSyncMode) {
    let code = match mode {
        LayerSyncMode::Auto => 0,
        LayerSyncMode::Off => 1,
        LayerSyncMode::On => 2,
    };
    synaptix_core::device::cuda::set_layer_sync_mode(code);
}

/// Trim CUDA-mempool после Drop KV-кэша. На non-CUDA сборках `hard_trim_all_pools_device`
/// резолвится в no-op fallback, поэтому функция доступна безусловно.
pub fn cuda_trim_pool(ordinal: i32) -> u64 {
    if ordinal < 0 {
        return 0;
    }
    let ord = ordinal as usize;
    let free_of = || {
        synaptix_core::device::cuda::mem_info(ord)
            .map(|(free, _total)| free)
            .unwrap_or(0)
    };
    let before = free_of();
    if let Err(e) = synaptix_core::memory::cuda_pool::hard_trim_all_pools_device(ord) {
        eprintln!("[synaptix] cuda_trim_pool({ord}): {e}");
        return 0;
    }
    (free_of().saturating_sub(before) / (1024 * 1024)) as u64
}

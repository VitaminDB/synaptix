//! Универсальный LLM-фасад: детект архитектуры → нативный pipeline (Qwen3 /
//! Qwen3-Next-Hybrid / Llama / Gemma3) за единым API. Токенайзер + jinja-шаблон
//! чата + восстановление текстовой дельты из id-токенов (DeltaSink) живут здесь,
//! чтобы и syn_chat, и CLI звали один и тот же путь. Добавление новой
//! LLM-архитектуры = новая ветка в [`LlmPipeline`]/[`load_llm`] — потребитель
//! не меняется.

use std::fmt;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex;

use synaptix_core::dtype::DType;
use synaptix_core::precision::{parse_dtype, PrecisionConfig};
use synaptix_core::tensor::Tensor;
use synaptix_llm_common::{
    GenerationConfig, KvCache as LlmKvCache, LinearSnapshot, StreamSink,
};
use synaptix_llm_gemma3::pipeline::GemmaPipeline;
use synaptix_llm_llama::pipeline::LlamaPipeline;
use synaptix_llm_muse_glimmer::pipeline::MusePipeline;
use synaptix_llm_muse_glimmer::DFlashCache;
use synaptix_llm_qwen3::pipeline::Qwen3Pipeline;
use synaptix_llm_qwen4_exp::pipeline::Qwen4ExpPipeline;
use synaptix_llm_qwen3_next_hybrid::pipeline::{HybridPipeline, MediaInput};
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
    /// MXFP8 (Blackwell block-scale): байт на элемент + E8M0-масштаб на блок
    /// из 32 → KV почти вдвое дешевле F16 (у гибрида 27B 33 КБ против 64 КБ на
    /// токен). Именно этот режим делает реальным контекст в 80k+ на 24 ГБ.
    MXFP8,
}

impl KvDtypePolicy {
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "half" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "f32" | "fp32" => Some(Self::F32),
            // `fp8e4m3` — имя из дропдауна настроек synthos; `fp8`/`mxfp8` —
            // как у CLI-флага `--kv-dtype`.
            "fp8e4m3" | "fp8_e4m3" | "fp8" | "mxfp8" => Some(Self::MXFP8),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::F16 => "f16",
            Self::BF16 => "bf16",
            Self::F32 => "f32",
            Self::MXFP8 => "fp8e4m3",
        }
    }

    fn to_dtype(self) -> DType {
        match self {
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::F32 => DType::F32,
            Self::MXFP8 => DType::MXFP8,
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

/// Настройки, при которых модель работает лучше всего. Считает их движок:
/// это он знает, какие пути у какой архитектуры выверены замерами, а
/// приложению остаётся их применить.
#[derive(Debug, Clone)]
pub struct OptimalProfile {
    pub policy: QuantPolicy,
    /// CUDA-graph на декоде.
    pub graph_decode: bool,
    /// Спекулятивный декод: MTP-голова у гибрида и Qwen4Exp, DFlash у
    /// Muse-Glimmer.
    pub speculation: bool,
    pub layer_sync: LayerSyncMode,
}

/// Выверенные настройки под архитектуру бандла.
///
/// Qwen4Exp держит KV квантованным: ядро по таблице блоков читает его
/// напрямую, отчего QSA на длинном промпте почти на треть быстрее, а памяти
/// под кэш нужно вдвое меньше. Спекуляция там же выключена — шаг упирается в
/// подкачку экспертов, и у второго токена почти свой их набор, так что
/// драфт только добавляет работы. У Muse-Glimmer наоборот: DFlash на
/// greedy-пути ничего не меняет в ответе и заметно ускоряет.
pub fn optimal_profile(path: &Path) -> OptimalProfile {
    let arch = crate::facade::arch::detect_llm_arch(path).ok();
    let mut policy = QuantPolicy::balance();
    let mut speculation = false;
    match arch {
        Some(crate::facade::arch::LlmArch::Qwen4Exp) => {
            policy.kv_dtype = KvDtypePolicy::MXFP8;
            policy.preset_name = "optimal".to_string();
        }
        Some(crate::facade::arch::LlmArch::MuseGlimmer) => {
            speculation = true;
            policy.preset_name = "optimal".to_string();
        }
        Some(crate::facade::arch::LlmArch::Hybrid) => {
            speculation = true;
            policy.preset_name = "optimal".to_string();
        }
        _ => policy.preset_name = "optimal".to_string(),
    }
    OptimalProfile {
        policy,
        graph_decode: false,
        speculation,
        layer_sync: LayerSyncMode::Auto,
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
    /// Имя инструмента для `role = "tool"`. Chat-шаблоны подписывают им блок
    /// результата (`<tool_output name="...">` у Muse Glimmer, `name` в
    /// ChatML-вариантах), поэтому без него модель не понимает, чей это вывод.
    pub name: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: "system".into(), content: content.into(), name: None }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), name: None }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: content.into(), name: None }
    }
    pub fn tool(content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into(), name: None }
    }

    /// Результат инструмента с именем вызванной функции.
    pub fn tool_named(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self { role: "tool".into(), content: content.into(), name: Some(name.into()) }
    }

    fn to_tok(&self) -> TokMessage {
        match self.role.as_str() {
            "system" => TokMessage::system(self.content.clone()),
            "assistant" => TokMessage::assistant(self.content.clone()),
            "tool" => TokMessage {
                role: MessageRole::Tool,
                content: self.content.clone(),
                name: self.name.clone(),
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
    Qwen4Exp(Qwen4ExpPipeline),
    Hybrid(HybridPipeline),
    Llama(LlamaPipeline),
    Gemma3(GemmaPipeline),
    MuseGlimmer(MusePipeline),
}

impl LlmPipeline {
    fn rope_capacity(&self) -> usize {
        match self {
            Self::Qwen3(p) => p.model.rope_capacity(),
            Self::Qwen4Exp(p) => p.rope_capacity(),
            Self::Hybrid(p) => p.model.rope_capacity(),
            Self::Llama(p) => p.model.rope_capacity(),
            Self::Gemma3(p) => p.model.rope_capacity(),
            Self::MuseGlimmer(p) => p.model.rope_capacity(),
        }
    }

    /// Отдать память карты, которую держат кэши модели (у Qwen4Exp это
    /// резидентные эксперты). Вызывается перед выгрузкой.
    fn release_device_caches(&self) {
        if let Self::Qwen4Exp(p) = self {
            p.release_device_caches();
        }
    }

    fn kv_bytes_per_token(&self) -> usize {
        match self {
            Self::Qwen3(p) => p.model.kv_bytes_per_token(),
            Self::Qwen4Exp(p) => p.kv_bytes_per_token(),
            Self::Hybrid(p) => p.model.kv_bytes_per_token(),
            Self::Llama(p) => p.model.kv_bytes_per_token(),
            Self::Gemma3(p) => p.model.kv_bytes_per_token(),
            Self::MuseGlimmer(p) => p.model.kv_bytes_per_token(),
        }
    }

    fn kv_fixed_bytes(&self, batch: usize, max_seq: usize) -> usize {
        match self {
            Self::Qwen3(p) => p.model.kv_fixed_bytes(batch, max_seq),
            Self::Qwen4Exp(p) => p.kv_fixed_bytes(batch, max_seq),
            Self::Hybrid(p) => p.model.kv_fixed_bytes(batch, max_seq),
            Self::Llama(p) => p.model.kv_fixed_bytes(batch, max_seq),
            Self::Gemma3(p) => p.model.kv_fixed_bytes(batch, max_seq),
            Self::MuseGlimmer(p) => p.model.kv_fixed_bytes(batch, max_seq),
        }
    }

    /// Генерация по промпту с медиа-вложениями (см. `LlmRunner::generate_streaming_media`).
    ///
    /// Обе мультимодальные архитектуры (Muse Glimmer, Qwen-гибрид) берут
    /// эмбеддинги склеенными по модальности в порядке появления в промпте
    /// и разбирают их курсорами по прогонам своих токенов-заполнителей.
    fn generate_streaming_media(
        &self,
        prompt_ids: &[u32],
        media: &[&MediaEmbedding],
        cfg: GenerationConfig,
        sink: &mut dyn StreamSink,
    ) -> Result<(), LlmError> {
        match self {
            LlmPipeline::MuseGlimmer(p) => {
                let images = concat_media(media, MediaKind::Image)?;
                let video = concat_media(media, MediaKind::Video)?;
                p.generate_with_mixed_media(prompt_ids, images.as_ref(), video.as_ref(), cfg, sink)
                    .map(|_| ())
                    .map_err(|e| LlmError(e.to_string()))
            }
            LlmPipeline::Hybrid(p) => {
                // По модальности: склейка эмбеддингов + сетка каждого блока
                // (у видео — одна сетка на каждую группу кадров) для M-RoPE.
                let mut inputs: Vec<MediaInput> = Vec::new();
                for (kind, pad) in [
                    (MediaKind::Image, p.config.image_token_id),
                    (MediaKind::Video, p.config.video_token_id),
                ] {
                    let Some(embeds) = concat_media(media, kind)? else { continue };
                    let pad = pad.ok_or_else(|| {
                        LlmError(format!("config.json без id заполнителя для {kind:?}"))
                    })?;
                    let grids: Vec<(usize, usize)> = media
                        .iter()
                        .filter(|m| m.kind == kind)
                        .flat_map(|m| std::iter::repeat_n(m.grid_hw, m.blocks))
                        .collect();
                    inputs.push(MediaInput { pad, embeds, grids });
                }
                p.generate_with_media(prompt_ids, &inputs, cfg, sink)
                    .map(|_| ())
                    .map_err(|e| LlmError(e.to_string()))
            }
            _ => Err(LlmError("архитектура не принимает медиа-вход".into())),
        }
    }

    /// Блок-заполнитель картинки в тексте промпта. Спецтокены у архитектур
    /// свои: Muse Glimmer — `<|image_start|><|patch|>…<|image_end|>`,
    /// Qwen-гибрид — `<|vision_start|><|image_pad|>…<|vision_end|>`.
    fn image_block(&self, tokens: usize) -> String {
        match self {
            LlmPipeline::Hybrid(_) => format!(
                "{QWEN_VISION_START_TOKEN}{}{QWEN_VISION_END_TOKEN}",
                QWEN_IMAGE_PAD_TOKEN.repeat(tokens)
            ),
            _ => format!(
                "{IMAGE_START_TOKEN}{}{IMAGE_END_TOKEN}",
                IMAGE_PAD_TOKEN.repeat(tokens)
            ),
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
            LlmPipeline::Qwen4Exp(p) => p
                .generate_streaming(prompt_ids, cfg, sink)
                .map(|_| ())
                .map_err(|e| LlmError(e.to_string())),
            LlmPipeline::MuseGlimmer(p) => {
                if cfg.temperature == 0.0 && dflash_enabled() && p.has_dflash() {
                    return p
                        .generate_dflash_streaming(prompt_ids, cfg, sink)
                        .map(|_| ())
                        .map_err(|e| LlmError(e.to_string()));
                }
                if cfg.temperature == 0.0 && matches!(p.model.device, Device::Cuda(_)) {
                    return p
                        .generate_lookup_streaming(prompt_ids, cfg, sink)
                        .map(|_| ())
                        .map_err(|e| LlmError(e.to_string()));
                }
                if p.graph_decode_supported() {
                    return p
                        .generate_with_graph_streaming(prompt_ids, cfg, sink)
                        .map(|_| ())
                        .map_err(|e| LlmError(e.to_string()));
                }
                p.generate_streaming(prompt_ids, cfg, sink)
                    .map(|_| ())
                    .map_err(|e| LlmError(e.to_string()))
            }
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

/// Персистентный KV-кэш диалога («префикс-KV»).
///
/// Держит посчитанный префикс и токены, которым он соответствует, поэтому ход
/// дописывает в кэш только НОВЫЙ хвост промпта, а не считает историю заново.
///
/// Точка возврата ставится на КОНЕЦ ПРОМПТА, а не на конец хода: chat-шаблон
/// перерисовывает реплику ассистента (у Qwen3 — `<think>` из промпта плюс
/// сгенерённый текст → `<think>\n\n</think>\n\n` + текст), так что
/// сгенерированные токены новому промпту префиксом не являются, а весь промпт
/// прошлого хода — является.
pub struct LlmKvSession {
    kv: LlmKvCache,
    ids: Vec<u32>,
    ctx_tokens: usize,
    kind: SessionKind,
}

/// Что, кроме основного KV, нужно донести до следующего хода.
enum SessionKind {
    /// Гибрид (Qwen3.6/3.8): кэш MTP-головы плюс снимок GDN-состояния — у
    /// linear-слоёв рекуррентность, её нельзя «откатить» одним `seq_len`.
    Hybrid {
        mtp_kv: LlmKvCache,
        snap: Option<Vec<LinearSnapshot>>,
        mtp_len: usize,
    },
    /// Muse-Glimmer: роллинг-окно контекста DFlash-драфтера. Linear-слоёв нет,
    /// поэтому снимок не нужен; зато у sliding-слоёв кэш держит лишь последние
    /// W токенов — граница должна попадать в окно.
    Muse { dcache: Option<DFlashCache> },
}

impl LlmKvSession {
    /// Сколько токенов контекста рассчитан кэш.
    pub fn ctx_tokens(&self) -> usize {
        self.ctx_tokens
    }

    /// Токенов в точке возврата.
    pub fn cached_tokens(&self) -> usize {
        self.ids.len()
    }

    /// Сколько токенов промпта уже посчитано.
    ///
    /// Ноль, если кэш не является полным префиксом промпта: частичное
    /// совпадение бесполезно — GDN-состояние на середину не откатить, а у
    /// sliding-слоёв ниже начала окна данных уже нет. Последний токен промпта
    /// всегда считается заново: из него берутся логиты первого шага.
    pub fn reusable(&self, prompt_ids: &[u32]) -> usize {
        let n = self.ids.len();
        if n == 0 || prompt_ids.len() <= n || prompt_ids[..n] != self.ids[..] {
            return 0;
        }
        match &self.kind {
            SessionKind::Hybrid { snap, .. } => {
                if snap.is_some() {
                    n
                } else {
                    0
                }
            }
            // Граница обязана лежать в ring-окне sliding-слоёв: всё, что ниже
            // их `start`, декод прошлого хода уже вытеснил.
            SessionKind::Muse { .. } => {
                if n >= self.kv.ring_start_max() {
                    n
                } else {
                    0
                }
            }
        }
    }

    /// Забыть префикс (смена чата/модели, сжатие истории).
    pub fn invalidate(&mut self) {
        self.ids.clear();
        self.kv.reset();
        match &mut self.kind {
            SessionKind::Hybrid { mtp_kv, snap, mtp_len } => {
                mtp_kv.reset();
                *snap = None;
                *mtp_len = 0;
            }
            SessionKind::Muse { dcache } => {
                if let Some(d) = dcache.as_mut() {
                    d.reset();
                }
            }
        }
    }

    /// Сбросить кэш к пустому состоянию перед полным префиллом.
    fn reset_for_full(&mut self) {
        self.kv.reset();
        match &mut self.kind {
            SessionKind::Hybrid { mtp_kv, snap, mtp_len } => {
                mtp_kv.reset();
                *snap = None;
                *mtp_len = 0;
            }
            SessionKind::Muse { dcache } => {
                if let Some(d) = dcache.as_mut() {
                    d.reset();
                }
            }
        }
    }
}

pub struct Llm {
    pipeline: Mutex<LlmPipeline>,
    config: LlmConfig,
    device: Device,
    vocab_size: usize,
    /// Путь к бандлу/каталогу модели — нужен для ленивой догрузки
    /// vision-башни уже после `load_llm` (см. [`Llm::ensure_media_tower`]).
    model_path: PathBuf,
    /// Compute-dtype, с которым загружены веса: тем же грузится и башня.
    compute_dtype: DType,
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

    /// Сессия префикс-KV на `ctx_tokens` токенов контекста и `max_new` токенов
    /// ответа (последнее нужно кэшу MTP-головы гибрида: он растёт быстрее
    /// основного).
    ///
    /// `Ok(None)` — архитектура пока не умеет продолжать с готового кэша.
    /// Поддержаны гибрид Qwen3.6/3.8 (с MTP) и Muse-Glimmer (включая
    /// DFlash-декод); остальные работают как раньше, без переиспользования.
    pub fn new_kv_session(
        &self,
        ctx_tokens: usize,
        max_new: usize,
    ) -> Result<Option<LlmKvSession>, LlmError> {
        let pipeline = self
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        let ctx = ctx_tokens.max(2);
        match &*pipeline {
            LlmPipeline::Hybrid(p) => {
                if !(mtp_enabled() && p.has_mtp()) {
                    return Ok(None);
                }
                let (kv, mtp_kv) = p
                    .make_mtp_caches(ctx, max_new)
                    .map_err(|e| LlmError(e.to_string()))?;
                Ok(Some(LlmKvSession {
                    kv,
                    ids: Vec::new(),
                    ctx_tokens: ctx,
                    kind: SessionKind::Hybrid { mtp_kv, snap: None, mtp_len: 0 },
                }))
            }
            LlmPipeline::MuseGlimmer(p) => {
                // Кэш драфтера нужен только greedy-пути DFlash; на прочих
                // (lookup / graph / eager) сессия — это просто основной KV.
                let (kv, dcache) = if p.has_dflash() {
                    let (kv, d) = p
                        .make_dflash_caches(ctx)
                        .map_err(|e| LlmError(e.to_string()))?;
                    (kv, Some(d))
                } else {
                    let kv = p
                        .model
                        .make_kv_cache(1, ctx)
                        .map_err(|e| LlmError(e.to_string()))?;
                    (kv, None)
                };
                Ok(Some(LlmKvSession {
                    kv,
                    ids: Vec::new(),
                    ctx_tokens: ctx,
                    kind: SessionKind::Muse { dcache },
                }))
            }
            _ => Ok(None),
        }
    }

    pub fn kv_bytes_per_token(&self) -> usize {
        self.pipeline
            .lock()
            .map(|p| p.kv_bytes_per_token())
            .unwrap_or(0)
    }

    /// Отдать память карты, занятую кэшами модели, не дожидаясь её `Drop`.
    pub fn release_device_caches(&self) {
        if let Ok(p) = self.pipeline.lock() {
            p.release_device_caches();
        }
    }

    /// VRAM под KV-слои постоянного размера (ring-окно sliding-слоёв) — их нет
    /// в ставке [`Self::kv_bytes_per_token`], но при аллокации кэша они
    /// занимают память. Бюджет контекста должен вычитать эту величину, иначе
    /// на sliding-моделях он завышен ровно на размер окон.
    pub fn kv_fixed_bytes(&self, max_seq: usize) -> usize {
        self.pipeline
            .lock()
            .map(|p| p.kv_fixed_bytes(1, max_seq))
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

// ── Мультимодальный вход: картинки и видео ──────────────────────────────────

/// Модальность вложения. Определяет, каким токеном-заполнителем вложение
/// представлено в промпте и в какой поток эмбеддингов оно попадёт.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

/// Закодированное vision-башней вложение.
///
/// `prompt_block` — готовая строка-заполнитель, которую вызывающий вставляет
/// в текст user-сообщения ДО применения chat-шаблона: у картинки это
/// `<|image_start|>` + `tokens` штук `<|patch|>` + `<|image_end|>`, у видео —
/// блок с таймкодами кадровых групп. Число заполнителей обязано совпасть с
/// числом строк в `embeds`, иначе генерация вернёт ошибку.
#[derive(Clone)]
pub struct MediaEmbedding {
    pub kind: MediaKind,
    pub embeds: Tensor,
    pub tokens: usize,
    pub prompt_block: String,
    /// Merged-сетка `(h, w)` одного блока (картинки / группы кадров) — для
    /// M-RoPE у Qwen-гибрида; `(0, 0)`, если архитектура её не использует.
    pub grid_hw: (usize, usize),
    /// Сколько блоков заполнителей в промпте: 1 у картинки, число групп
    /// кадров у видео.
    pub blocks: usize,
}

impl MediaEmbedding {
    /// Сколько токенов контекста займёт вложение.
    pub fn tokens(&self) -> usize {
        self.tokens
    }
}

impl Llm {
    /// Умеет ли архитектура принимать медиа-вход и есть ли это в сборке:
    /// у Muse Glimmer — `vision_config` в конфиге, у Qwen-гибрида
    /// (Qwen3.5/3.6/3.8) — `image_token_id` в конфиге и компонент башни
    /// в бандле (`model.visual.*`). Иначе это text-only.
    pub fn supports_media(&self) -> bool {
        match self.pipeline.lock() {
            Ok(p) => match &*p {
                LlmPipeline::MuseGlimmer(m) => m.config.vision.is_some(),
                LlmPipeline::Hybrid(h) => {
                    h.config.image_token_id.is_some()
                        && HybridPipeline::bundle_has_vision(&self.model_path)
                }
                _ => false,
            },
            Err(_) => false,
        }
    }

    /// Загружена ли vision-башня в память устройства.
    pub fn media_tower_loaded(&self) -> bool {
        match self.pipeline.lock() {
            Ok(p) => match &*p {
                LlmPipeline::MuseGlimmer(m) => m.has_vision(),
                LlmPipeline::Hybrid(h) => h.has_vision(),
                _ => false,
            },
            Err(_) => false,
        }
    }

    /// Идемпотентно догружает vision-башню из того же бандла, что и веса LLM.
    ///
    /// `Ok(false)` — в бандле нет тензоров башни (text-only сборка). Башня
    /// стоит ощутимой VRAM, поэтому вызывающий обычно грузит её на время
    /// кодирования вложений и сразу отпускает через [`Self::release_media_tower`].
    pub fn ensure_media_tower(&self) -> Result<bool, LlmError> {
        let mut guard = self
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        match &mut *guard {
            LlmPipeline::MuseGlimmer(m) => {
                if m.has_vision() {
                    return Ok(true);
                }
                m.load_vision(&self.model_path, self.compute_dtype)
                    .map_err(|e| LlmError(format!("vision load: {e}")))
            }
            LlmPipeline::Hybrid(h) => {
                if h.has_vision() {
                    return Ok(true);
                }
                h.load_vision(&self.model_path, self.compute_dtype)
                    .map_err(|e| LlmError(format!("vision load: {e}")))
            }
            _ => Ok(false),
        }
    }

    /// Выгружает vision-башню и возвращает её память пулу устройства.
    pub fn release_media_tower(&self) {
        if let Ok(mut guard) = self.pipeline.lock() {
            match &mut *guard {
                LlmPipeline::MuseGlimmer(m) => m.release_vision(),
                LlmPipeline::Hybrid(h) => h.release_vision(),
                _ => {}
            }
        }
    }

    /// Кодирует картинку в эмбеддинги vision-башни.
    ///
    /// `max_tokens` ограничивает число vision-токенов сверху (даунскейл в
    /// препроцессинге). `None` — потолок из конфига модели (у Muse Glimmer 4096).
    pub fn encode_image(
        &self,
        path: &Path,
        max_tokens: Option<usize>,
    ) -> Result<MediaEmbedding, LlmError> {
        let guard = self
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        let (embeds, grid_hw) = match &*guard {
            LlmPipeline::MuseGlimmer(m) => (
                m.encode_image_limited(path, max_tokens)
                    .map_err(|e| LlmError(format!("image encode: {e}")))?,
                (0, 0),
            ),
            LlmPipeline::Hybrid(h) => h
                .encode_image_limited_with_grid(path, max_tokens)
                .map_err(|e| LlmError(format!("image encode: {e}")))?,
            _ => return Err(LlmError("архитектура не принимает картинки".into())),
        };
        let tokens = embeds.dims()[0];
        let prompt_block = guard.image_block(tokens);
        Ok(MediaEmbedding { kind: MediaKind::Image, embeds, tokens, prompt_block, grid_hw, blocks: 1 })
    }

    /// Кодирует видео: сэмплинг кадров (ffmpeg) → vision-башня. Блок промпта
    /// несёт таймкоды кадровых групп, поэтому модель видит временну́ю шкалу.
    pub fn encode_video(&self, path: &Path) -> Result<MediaEmbedding, LlmError> {
        let guard = self
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        let (embeds, prompt_block, grid_hw, blocks) = match &*guard {
            LlmPipeline::MuseGlimmer(m) => {
                let (embeds, info) = m
                    .encode_video(path)
                    .map_err(|e| LlmError(format!("video encode: {e}")))?;
                (embeds, info.prompt_block(), (0, 0), info.groups)
            }
            LlmPipeline::Hybrid(h) => {
                let (embeds, info) = h
                    .encode_video(path)
                    .map_err(|e| LlmError(format!("video encode: {e}")))?;
                (embeds, info.prompt_block(), info.grid_hw, info.groups)
            }
            _ => return Err(LlmError("архитектура не принимает видео".into())),
        };
        let tokens = embeds.dims()[0];
        Ok(MediaEmbedding {
            kind: MediaKind::Video,
            embeds,
            tokens,
            prompt_block,
            grid_hw,
            blocks,
        })
    }
}

/// Обёртки блока картинки в промпте Muse Glimmer (см. `MediaEmbedding::prompt_block`).
const IMAGE_START_TOKEN: &str = "<|image_start|>";
const IMAGE_END_TOKEN: &str = "<|image_end|>";
/// Заполнитель одного vision-токена картинки — `config.image_token_id`.
const IMAGE_PAD_TOKEN: &str = "<|patch|>";
/// То же для Qwen-гибрида (Qwen3.5/3.6/3.8 — токены семейства Qwen-VL).
const QWEN_VISION_START_TOKEN: &str = "<|vision_start|>";
const QWEN_VISION_END_TOKEN: &str = "<|vision_end|>";
const QWEN_IMAGE_PAD_TOKEN: &str = "<|image_pad|>";

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
                // Канальные шаблоны (Muse Glimmer) не знают про
                // `enable_thinking`: у них глубина рассуждений задаётся
                // строкой `reasoning_strength` в системном блоке, а совсем
                // отключить reasoning протокол не позволяет. Отдаём обе
                // переменные — лишнюю шаблон просто не прочитает.
                let strength = if enable_thinking { "high" } else { "low" };
                let mut opts = RenderOptions::new()
                    .with_generation_prompt(add_generation_prompt)
                    .with_var("enable_thinking", serde_json::Value::Bool(enable_thinking))
                    .with_var("reasoning_strength", serde_json::Value::String(strength.into()));
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
            repeat_last_n: self.opts.repeat_last_n,
            presence_penalty: self.opts.presence_penalty,
            frequency_penalty: self.opts.frequency_penalty,
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

impl<'a> LlmGeneration<'a> {
    /// Как [`Self::generate_streaming`], но с префикс-KV: всё, что уже лежит в
    /// `session`, не считается заново. Возвращает, сколько токенов промпта
    /// удалось переиспользовать.
    ///
    /// Сессия обновляется на КОНЕЦ ПРОМПТА этого хода — следующий ход,
    /// дописавший в историю ответ модели и результаты инструментов, продолжит с
    /// этой точки. Расхождение промпта с кэшем (сжали историю, отредактировали
    /// сообщение, сменили системный prompt) обнаруживается сравнением токенов и
    /// просто приводит к полному префиллу.
    pub fn generate_streaming_cached<F>(
        &mut self,
        session: &mut LlmKvSession,
        prompt_ids: &[u32],
        tokenizer: &LlmTokenizer,
        on_token: F,
    ) -> Result<usize, LlmError>
    where
        F: FnMut(u32, &str) -> bool,
    {
        let pipeline = self
            .model
            .pipeline
            .lock()
            .map_err(|_| LlmError("pipeline mutex poisoned".into()))?;
        let cfg = GenerationConfig {
            max_new_tokens: self.opts.max_new_tokens,
            temperature: self.opts.temperature,
            top_k: self.opts.top_k,
            top_p: self.opts.top_p,
            min_p: self.opts.min_p,
            repetition_penalty: self.opts.repeat_penalty,
            repeat_last_n: self.opts.repeat_last_n,
            presence_penalty: self.opts.presence_penalty,
            frequency_penalty: self.opts.frequency_penalty,
            seed: self.opts.seed,
            eos_token_id: None,
            eos_token_ids: self.stop_tokens.clone(),
            max_seq: Some(session.ctx_tokens),
            prefill_batch: prefill_chunk_size(),
        };
        let mut sink = DeltaSink {
            tokenizer,
            on_token,
            acc: Vec::new(),
            decoded: String::new(),
            stop_sequences: &self.stop_sequences,
        };

        match &*pipeline {
            LlmPipeline::Hybrid(p) => {
                if !matches!(session.kind, SessionKind::Hybrid { .. }) {
                    return Err(LlmError("префикс-KV: сессия от другой модели".into()));
                }
                // Сравнивать префиксы нужно в том виде, в каком промпт видит движок.
                let ids = p.maybe_prepend_bos(prompt_ids);
                let reuse = session.reusable(&ids);
                if reuse > 0 {
                    session.kv.seq_len = reuse;
                    let SessionKind::Hybrid { mtp_kv, snap, mtp_len } = &mut session.kind else {
                        unreachable!("проверено выше")
                    };
                    if let Some(sn) = snap.as_ref() {
                        session
                            .kv
                            .restore_linear(sn)
                            .map_err(|e| LlmError(e.to_string()))?;
                    }
                    mtp_kv.seq_len = *mtp_len;
                } else {
                    // Полный префилл: GDN-состояние обязано быть нулевым, иначе
                    // ход продолжит чужую рекуррентность.
                    session.reset_for_full();
                }

                let LlmKvSession { kv, ids: sess_ids, kind, .. } = session;
                let SessionKind::Hybrid { mtp_kv, snap, mtp_len } = kind else {
                    unreachable!("проверено выше")
                };
                let mut new_snap: Option<Vec<LinearSnapshot>> = None;
                let mut new_mtp_len = 0usize;
                let mut new_len = 0usize;
                let res = p.generate_mtp_resume(
                    kv,
                    mtp_kv,
                    &ids,
                    cfg,
                    &mut sink,
                    &mut |at: usize, kv_at: &LlmKvCache, mtp_at: &LlmKvCache| {
                        new_snap = Some(kv_at.snapshot_linear_full()?);
                        new_mtp_len = mtp_at.seq_len;
                        new_len = at;
                        Ok(())
                    },
                );
                // Точку возврата обновляем, только если движок её действительно
                // снял (он ставит её на границу, кратную чанку GDN-скана, —
                // короткий промпт такой границы может не иметь). Иначе прежняя
                // остаётся в силе: она по-прежнему префикс этого промпта. Это
                // верно и при ошибке генерации — хвост за точкой всё равно
                // перезапишется следующим ходом.
                if let Some(sn) = new_snap {
                    *snap = Some(sn);
                    *mtp_len = new_mtp_len;
                    *sess_ids = ids[..new_len].to_vec();
                }
                match res {
                    Ok(_) => Ok(reuse),
                    Err(e) => Err(LlmError(e.to_string())),
                }
            }
            LlmPipeline::MuseGlimmer(p) => {
                if !matches!(session.kind, SessionKind::Muse { .. }) {
                    return Err(LlmError("префикс-KV: сессия от другой модели".into()));
                }
                let ids = p.maybe_prepend_bos(prompt_ids);
                let mut reuse = session.reusable(&ids);
                // Путь декода выбирается так же, как в `generate_streaming`.
                let dflash_path =
                    cfg.temperature == 0.0 && dflash_enabled() && p.has_dflash();
                let SessionKind::Muse { dcache } = &mut session.kind else {
                    unreachable!("проверено выше")
                };
                if reuse > 0 && dflash_path {
                    // Контекст драфтера — роллинг-окно: если граница из него
                    // уже вышла, набирать его заново нечем (tap-hidden'ы
                    // префикса не хранятся), поэтому честно префиллим всё.
                    let ok = dcache
                        .as_mut()
                        .map(|d| d.truncate_to(reuse))
                        .unwrap_or(false);
                    if !ok {
                        reuse = 0;
                    }
                }
                if reuse > 0 {
                    session.kv.seq_len = reuse;
                } else {
                    session.reset_for_full();
                }
                let LlmKvSession { kv, ids: sess_ids, kind, .. } = session;
                let SessionKind::Muse { dcache } = kind else {
                    unreachable!("проверено выше")
                };
                // Порядок ровно как в `LlmPipeline::generate_streaming`: иначе
                // ход с кэшем и ход без кэша разрешали бы greedy-ничьи разными
                // путями (спекулятивный против обычного) и расходились в тексте.
                let res = if dflash_path {
                    let d = dcache
                        .as_mut()
                        .ok_or_else(|| LlmError("префикс-KV: сессия без кэша DFlash".into()))?;
                    p.generate_dflash_resume(kv, d, &ids, cfg, &mut sink)
                        .map(|_| ())
                } else if cfg.temperature == 0.0
                    && matches!(p.model.device, synaptix_core::device::Device::Cuda(_))
                {
                    p.generate_lookup_resume(kv, &ids, cfg, &mut sink).map(|_| ())
                } else if p.graph_decode_supported() {
                    p.generate_with_graph_resume(kv, &ids, cfg, &mut sink).map(|_| ())
                } else {
                    p.generate_streaming_resume(kv, &ids, cfg, &mut sink).map(|_| ())
                };
                // У Muse точка возврата — весь промпт: linear-слоёв нет, а
                // attention-кэш усекается по `seq_len` (в пределах ring-окна).
                *sess_ids = ids;
                match res {
                    Ok(()) => Ok(reuse),
                    Err(e) => Err(LlmError(e.to_string())),
                }
            }
            _ => Err(LlmError("префикс-KV: архитектура не поддержана".into())),
        }
    }

    /// Стриминг по промпту с медиа-вложениями.
    ///
    /// `media` перечисляется **в том же порядке, в каком блоки-заполнители
    /// стоят в промпте**: эмбеддинги одной модальности склеиваются по этому
    /// порядку и разбираются курсором по прогонам её токена-заполнителя.
    /// Спекулятивные пути (DFlash / lookup / CUDA-graph) на медиа-промпте не
    /// применяются — prefill идёт по готовым эмбеддингам.
    pub fn generate_streaming_media<F>(
        &mut self,
        prompt_ids: &[u32],
        tokenizer: &LlmTokenizer,
        media: &[&MediaEmbedding],
        on_token: F,
    ) -> Result<(), LlmError>
    where
        F: FnMut(u32, &str) -> bool,
    {
        if media.is_empty() {
            return Err(LlmError("generate_streaming_media без вложений".into()));
        }

        let cfg = GenerationConfig {
            max_new_tokens: self.opts.max_new_tokens,
            temperature: self.opts.temperature,
            top_k: self.opts.top_k,
            top_p: self.opts.top_p,
            min_p: self.opts.min_p,
            repetition_penalty: self.opts.repeat_penalty,
            repeat_last_n: self.opts.repeat_last_n,
            presence_penalty: self.opts.presence_penalty,
            frequency_penalty: self.opts.frequency_penalty,
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
        pipeline.generate_streaming_media(prompt_ids, media, cfg, &mut sink)
    }
}

/// Склеивает эмбеддинги одной модальности в порядке следования вложений.
/// `None` — вложений этой модальности нет.
fn concat_media(media: &[&MediaEmbedding], kind: MediaKind) -> Result<Option<Tensor>, LlmError> {
    let parts: Vec<&Tensor> = media
        .iter()
        .filter(|m| m.kind == kind)
        .map(|m| &m.embeds)
        .collect();
    match parts.len() {
        0 => Ok(None),
        1 => Ok(Some(parts[0].clone())),
        _ => Tensor::cat(&parts, 0)
            .map(Some)
            .map_err(|e| LlmError(format!("media cat: {e}"))),
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
        let delta = stream_delta(full, &mut self.decoded);

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

/// Дельта стрима: `full` — декод ВСЕХ накопленных токенов, `decoded` — что уже
/// ушло наружу.
///
/// Байтовый BPE режет многобайтовые символы (эмодзи, CJK) на несколько
/// токенов, и декод частичной последовательности ставит в хвост U+FFFD «�».
/// Раньше этот «�» уходил дельтой в UI, а на следующем токене
/// `full.starts_with(decoded)` не сходился («😀» ≠ «�») и настоящий символ
/// молча терялся — в чате эмодзи просто пропадали. Поэтому недописанный хвост
/// придерживаем: `decoded` не двигается, дельта пустая, символ уйдёт целиком
/// со следующим токеном.
fn stream_delta(full: String, decoded: &mut String) -> String {
    if full.ends_with('\u{FFFD}') {
        return String::new();
    }
    let delta = if full.len() >= decoded.len() && full.starts_with(decoded.as_str()) {
        full[decoded.len()..].to_string()
    } else {
        // Ресинк: декод «переписал» уже показанный текст (смена нормализации
        // токенайзера и т. п.) — дельту не выдумываем, просто догоняем.
        String::new()
    };
    *decoded = full;
    delta
}

#[cfg(test)]
mod stream_delta_tests {
    use super::stream_delta;

    /// Прогоняет последовательность decode-снапшотов через stream_delta.
    fn run(fulls: &[&str]) -> (Vec<String>, String) {
        let mut decoded = String::new();
        let deltas = fulls
            .iter()
            .map(|f| stream_delta(f.to_string(), &mut decoded))
            .collect();
        (deltas, decoded)
    }

    #[test]
    fn plain_text_streams_as_diffs() {
        let (deltas, decoded) = run(&["При", "Привет", "Привет!"]);
        assert_eq!(deltas, vec!["При", "вет", "!"]);
        assert_eq!(decoded, "Привет!");
    }

    #[test]
    fn split_emoji_is_held_back_then_emitted_whole() {
        // Эмодзи разрезан BPE: decode частичной последовательности даёт «�».
        let (deltas, decoded) = run(&["Привет ", "Привет \u{FFFD}", "Привет 😀"]);
        assert_eq!(deltas, vec!["Привет ", "", "😀"]);
        assert_eq!(decoded, "Привет 😀");
    }

    #[test]
    fn multi_token_emoji_with_double_replacement() {
        // 4-байтовый эмодзи может ехать три токена: «��» в хвосте оба раза.
        let (deltas, _) = run(&["ок ", "ок \u{FFFD}", "ок \u{FFFD}\u{FFFD}", "ок 🎉"]);
        assert_eq!(deltas, vec!["ок ", "", "", "🎉"]);
    }

    #[test]
    fn resync_after_rewrite_does_not_invent_delta() {
        let (deltas, decoded) = run(&["abc", "xyz"]);
        assert_eq!(deltas, vec!["abc", ""]);
        assert_eq!(decoded, "xyz");
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
    compute_dtype: DType,
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
        model_path: path.to_path_buf(),
        compute_dtype,
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
        LlmArch::Qwen4Exp => {
            Qwen4ExpPipeline::load_with_precision(path, device, precision, max_seq)
                .map(LlmPipeline::Qwen4Exp)
                .map_err(|e| LlmError(format!("load qwen4_exp: {e}")))?
        }
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
            let mut p = MusePipeline::load_with_precision(path, device, precision, max_seq)
                .map_err(|e| LlmError(format!("load muse_glimmer: {e}")))?;
            if dflash_enabled() {
                match p.load_dflash(path, precision) {
                    Ok(true) => eprintln!("[synaptix] muse_glimmer: DFlash-драфтер подключён"),
                    Ok(false) => {}
                    Err(e) => eprintln!("[synaptix] muse_glimmer: DFlash пропущен: {e}"),
                }
            }
            LlmPipeline::MuseGlimmer(p)
        }
    };

    build_facade(pipeline, path, device, max_seq, precision.compute)
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
    // Веса (и транзиенты деквантизации/переквантизации) — в default-пул,
    // отдельно от пула активаций: смешанный пул за один длинный префилл
    // деградирует в «решето» и упирается в OOM при неизменном живом объёме
    // (см. `synaptix_core::device::cuda::activations_pool`).
    let _weights = synaptix_core::device::cuda::WeightsAllocGuard::for_device(*device);
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

/// DFlash — блочный спекулятивный декод Muse-Glimmer на драфтере-ассистенте
/// (компонент `dflash` в бандле). Аналог MTP у гибрида: greedy-путь lossless.
static DFLASH_ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(true);

pub fn set_dflash_enabled(on: bool) {
    DFLASH_ENABLED.store(on, std::sync::atomic::Ordering::Relaxed);
}

pub fn dflash_enabled() -> bool {
    DFLASH_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
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

/// Отдать драйверу device-кэши CUDA-ядер (TMA-дескрипторы, MXFP8-скретчи).
/// Они живут в статиках и переживают выгрузку модели: мелкие, но рассыпаны по
/// сегментам mempool'а и не дают [`cuda_trim_pool`] вернуть зарезервированное.
/// Возвращает (сброшено дескрипторов, освобождено байт скретчей).
pub fn cuda_release_kernel_caches() -> (usize, usize) {
    synaptix_kernels_cuda::release_device_caches()
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

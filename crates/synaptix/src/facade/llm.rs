//! Языковые модели: токенизатор и chat-шаблон модели, сообщения, профиль
//! сэмплинга — без весов (клиенту службы — считать токены и собирать
//! историю так же, как служба). Загрузка и генерация — [`engine`] (фича
//! `engine`): детект архитектуры → нативный pipeline за единым API.

use std::fmt;
use std::path::Path;

use synaptix_tokenizer::templates::chat_template::RenderOptions;
use synaptix_tokenizer::templates::tools::{ToolCall as TokToolCall, ToolCallFunction as TokToolCallFunction};
use synaptix_tokenizer::{ChatTemplate, HfTokenizer, Message as TokMessage, MessageRole, SpecialTokenKind, SpecialTokens, Tokenizer as _};

use super::arch::{detect_llm_arch, read_model_file};

pub use super::sampling::{sampling_profile, ReasoningLevels, ReasoningVar, SamplingPreset, SamplingProfile};

#[cfg(feature = "engine")]
mod engine;
#[cfg(feature = "engine")]
pub use engine::*;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Имя инструмента для `role = "tool"`. Chat-шаблоны подписывают им блок
    /// результата (`<tool_output name="...">` у Muse Glimmer, `name` в
    /// ChatML-вариантах), поэтому без него модель не понимает, чей это вывод.
    pub name: Option<String>,
    /// Размышления реплики ассистента — шаблон рендерит их сам
    /// (`reasoning_content` у Qwen3.x). `None` — без блока размышлений.
    pub reasoning: Option<String>,
    /// Вызовы инструментов реплики ассистента: (имя, JSON аргументов).
    /// Шаблон рендерит их в своём родном формате (у Qwen3.8 —
    /// `<function=…><parameter=…>`), поэтому в `content` их класть не надо.
    pub tool_calls: Vec<(String, String)>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self::plain("system", content, None)
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self::plain("user", content, None)
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self::plain("assistant", content, None)
    }
    pub fn tool(content: impl Into<String>) -> Self {
        Self::plain("tool", content, None)
    }

    /// Результат инструмента с именем вызванной функции.
    pub fn tool_named(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self::plain("tool", content, Some(name.into()))
    }

    /// Реплика ассистента целиком: текст, размышления и структурные вызовы.
    pub fn assistant_turn(
        content: impl Into<String>,
        reasoning: Option<String>,
        tool_calls: Vec<(String, String)>,
    ) -> Self {
        Self {
            role: "assistant".into(),
            content: content.into(),
            name: None,
            reasoning,
            tool_calls,
        }
    }

    fn plain(role: &str, content: impl Into<String>, name: Option<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            name,
            reasoning: None,
            tool_calls: Vec::new(),
        }
    }

    pub(crate) fn to_tok(&self) -> TokMessage {
        match self.role.as_str() {
            "system" => TokMessage::system(self.content.clone()),
            "assistant" => TokMessage::assistant_with_reasoning(
                self.content.clone(),
                self.reasoning.clone(),
                self.tool_calls
                    .iter()
                    .map(|(name, arguments)| TokToolCall {
                        id: None,
                        call_type: "function".into(),
                        function: TokToolCallFunction {
                            name: name.clone(),
                            arguments: arguments.clone(),
                        },
                    })
                    .collect(),
            ),
            "tool" => TokMessage {
                role: MessageRole::Tool,
                content: self.content.clone(),
                name: self.name.clone(),
                tool_call_id: None,
                tool_calls: Vec::new(),
                reasoning_content: None,
            },
            _ => TokMessage::user(self.content.clone()),
        }
    }
}

pub struct LlmTokenizer {
    pub(crate) tokenizer: HfTokenizer,
    pub(crate) template: Option<ChatTemplate>,
    /// Уровни глубины размышлений, которые понимает шаблон (см.
    /// [`super::sampling::reasoning_levels`]).
    pub(crate) reasoning: Option<ReasoningLevels>,
    pub(crate) eos_ids: Vec<u32>,
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
        self.apply_chat_template_reasoning(messages, add_generation_prompt, enable_thinking, None, tools)
    }

    /// Уровни глубины размышлений, которые понимает шаблон модели.
    pub fn reasoning_levels(&self) -> Option<&ReasoningLevels> {
        self.reasoning.as_ref()
    }

    /// Как [`Self::apply_chat_template_ex_tools`], но с глубиной размышлений:
    /// `effort` — уровень из [`Self::reasoning_levels`]. `None` или уровень,
    /// которого шаблон не знает, — уровень шаблона по умолчанию.
    pub fn apply_chat_template_reasoning(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
        enable_thinking: bool,
        effort: Option<&str>,
        tools: Option<&[serde_json::Value]>,
    ) -> Result<String, LlmError> {
        let msgs: Vec<TokMessage> = messages.iter().map(Message::to_tok).collect();
        match &self.template {
            Some(tmpl) => {
                let mut opts = super::sampling::with_thinking_vars(
                    RenderOptions::new().with_generation_prompt(add_generation_prompt),
                    self.reasoning.as_ref(),
                    enable_thinking,
                    effort,
                );
                // Как `apply_chat_template` в transformers: `tools` определён
                // всегда, без инструментов — `none`. Шаблон Llama-3.2 проверяет
                // `tools is not none`, а у неопределённой переменной это
                // истина — модель получала инструкцию «ответь JSON-вызовом».
                match tools {
                    Some(t) if !t.is_empty() => {
                        opts = opts.with_var("tools", serde_json::Value::Array(t.to_vec()));
                    }
                    _ => opts = opts.with_var("tools", serde_json::Value::Null),
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

/// Источник jinja-шаблона: `chat_template.jinja` либо
/// `tokenizer_config.json["chat_template"]`. None → fallback ChatML.
pub(crate) fn load_template_source(model: &Path) -> Option<String> {
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

pub(crate) fn build_tokenizer(model: &Path) -> Result<HfTokenizer, LlmError> {
    let bytes = read_model_file(model, "tokenizer.json")
        .ok_or_else(|| LlmError(format!("tokenizer.json не найден в {}", model.display())))?;
    HfTokenizer::from_bytes(&bytes).map_err(|e| LlmError(format!("tokenizer.json: {e}")))
}

/// Стоп-токены: eos из specials, конец хода чата (ChatML `<|im_end|>`,
/// Gemma-3 `<end_of_turn>`, Gemma-4 `<turn|>`) и весь список из
/// `generation_config.json` — там он у некоторых моделей шире, чем в
/// `config.json`, а стоп-токен «конец хода» живёт только там.
pub(crate) fn build_eos_ids(tok: &HfTokenizer, specials: &SpecialTokens, model: &Path) -> Vec<u32> {
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
    push(tok.token_to_id("<end_of_turn>"));
    push(tok.token_to_id("<turn|>"));
    for name in ["generation_config.json", "config.json"] {
        let Some(bytes) = read_model_file(model, name) else { continue };
        let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) else { continue };
        match v.get("eos_token_id") {
            Some(serde_json::Value::Array(a)) => {
                for x in a {
                    push(x.as_u64().map(|x| x as u32));
                }
            }
            Some(serde_json::Value::Number(n)) => push(n.as_u64().map(|x| x as u32)),
            _ => {}
        }
    }
    ids
}

/// Токенизатор и chat-шаблон модели без весов: клиенту службы — считать
/// токены и собирать промпт так же, как служба.
pub fn load_llm_tokenizer(path: &Path) -> Result<LlmTokenizer, LlmError> {
    let tokenizer = build_tokenizer(path)?;
    let specials = tokenizer.special_tokens().clone();
    let eos_ids = build_eos_ids(&tokenizer, &specials, path);
    let template = load_template_source(path).map(|src| ChatTemplate::from_source_with_specials(src, specials.clone()));
    let reasoning = template.as_ref().and_then(|t| super::sampling::reasoning_levels(detect_llm_arch(path).ok(), t.source()));
    Ok(LlmTokenizer { tokenizer, template, reasoning, eos_ids })
}


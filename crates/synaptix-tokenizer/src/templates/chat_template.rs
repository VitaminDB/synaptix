use std::collections::BTreeMap;

use minijinja::value::Value as JjValue;
use minijinja::context;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

use crate::error::Result;
use crate::special_tokens::SpecialTokens;
use crate::templates::jinja::JinjaEnv;
use crate::templates::reasoning::ReasoningConfig;
use crate::templates::tools::{ToolCall, ToolDef};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MessageRole {
    System,
    User,
    Assistant,
    Tool,
}

impl MessageRole {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::System => "system",
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: MessageRole,
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: MessageRole::System, content: content.into(), name: None, tool_call_id: None, tool_calls: Vec::new() }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self { role: MessageRole::User, content: content.into(), name: None, tool_call_id: None, tool_calls: Vec::new() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: MessageRole::Assistant, content: content.into(), name: None, tool_call_id: None, tool_calls: Vec::new() }
    }

    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: MessageRole::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
            tool_calls: Vec::new(),
        }
    }

    pub fn assistant_with_tools(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
            tool_calls: calls,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RenderOptions {
    pub add_generation_prompt: bool,
    pub tools: Vec<ToolDef>,
    pub extra_vars: BTreeMap<String, JsonValue>,
    pub reasoning: Option<ReasoningConfig>,
}

impl RenderOptions {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_generation_prompt(mut self, v: bool) -> Self {
        self.add_generation_prompt = v;
        self
    }

    pub fn with_tools(mut self, tools: Vec<ToolDef>) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_var(mut self, k: impl Into<String>, v: JsonValue) -> Self {
        self.extra_vars.insert(k.into(), v);
        self
    }

    pub fn with_reasoning(mut self, r: ReasoningConfig) -> Self {
        self.reasoning = Some(r);
        self
    }
}

#[derive(Clone)]
pub struct ChatTemplate {
    env: JinjaEnv,
    source: String,
    specials: SpecialTokens,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate")
            .field("source.len", &self.source.len())
            .field("specials", &self.specials)
            .finish()
    }
}

impl ChatTemplate {
    pub fn from_source(source: impl Into<String>) -> Self {
        Self { env: JinjaEnv::new(), source: source.into(), specials: SpecialTokens::default() }
    }

    pub fn from_source_with_specials(source: impl Into<String>, specials: SpecialTokens) -> Self {
        Self { env: JinjaEnv::new(), source: source.into(), specials }
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn render(&self, msgs: &[Message], opts: &RenderOptions) -> Result<String> {
        let messages_for_template = self.prepare_messages(msgs, opts);
        let messages_json: Vec<JsonValue> = messages_for_template
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<_, _>>()?;
        let tools_json: Vec<JsonValue> = opts
            .tools
            .iter()
            .map(serde_json::to_value)
            .collect::<std::result::Result<_, _>>()?;
        let specials_obj: BTreeMap<String, String> = self
            .specials
            .iter()
            .map(|(k, slot)| (format!("{:?}", k).to_lowercase(), slot.token.clone()))
            .collect();
        let mut ctx_obj = serde_json::Map::new();
        ctx_obj.insert("messages".into(), JsonValue::Array(messages_json));
        ctx_obj.insert("add_generation_prompt".into(), JsonValue::Bool(opts.add_generation_prompt));
        ctx_obj.insert("tools".into(), JsonValue::Array(tools_json));
        if let Some(slot) = self.specials.get(crate::SpecialTokenKind::Bos) {
            ctx_obj.insert("bos_token".into(), JsonValue::String(slot.token.clone()));
        }
        if let Some(slot) = self.specials.get(crate::SpecialTokenKind::Eos) {
            ctx_obj.insert("eos_token".into(), JsonValue::String(slot.token.clone()));
        }
        if let Some(slot) = self.specials.get(crate::SpecialTokenKind::Pad) {
            ctx_obj.insert("pad_token".into(), JsonValue::String(slot.token.clone()));
        }
        if let Some(slot) = self.specials.get(crate::SpecialTokenKind::Unk) {
            ctx_obj.insert("unk_token".into(), JsonValue::String(slot.token.clone()));
        }
        ctx_obj.insert("special_tokens".into(), JsonValue::Object(
            specials_obj.into_iter().map(|(k, v)| (k, JsonValue::String(v))).collect(),
        ));
        for (k, v) in &opts.extra_vars {
            ctx_obj.insert(k.clone(), v.clone());
        }
        let ctx_value = JsonValue::Object(ctx_obj);
        let jj_ctx: JjValue = serde_json::from_value::<JjValue>(ctx_value)
            .unwrap_or_else(|_| context! {});
        self.env.render(&self.source, jj_ctx)
    }

    fn prepare_messages(&self, msgs: &[Message], opts: &RenderOptions) -> Vec<Message> {
        let Some(reasoning) = opts.reasoning.as_ref() else {
            return msgs.to_vec();
        };
        if !reasoning.strip_reasoning_in_history {
            return msgs.to_vec();
        }
        msgs.iter()
            .map(|m| {
                if m.role == MessageRole::Assistant {
                    let stripped = reasoning.strip(&m.content);
                    Message { content: stripped, ..m.clone() }
                } else {
                    m.clone()
                }
            })
            .collect()
    }
}

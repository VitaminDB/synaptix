pub mod channel;
pub mod json_repair;
pub mod think;
pub mod tool_parser;

use synaptix_tokenizer::{Gemma4Ids, Gemma4StreamParser};

use crate::facade::llm::LlmTokenizer;
use channel::{ChannelIds, ChannelParser};
use think::ThinkParser;
pub use tool_parser::{LlamaJsonParser, RawToolCall, ToolCallParser};

pub enum StreamParser {
    ChatML { think: ThinkParser, tools: ToolCallParser },
    Channel(ChannelParser),
    Gemma(Gemma4StreamParser),
    Llama(LlamaJsonParser),
}

impl StreamParser {
    pub fn for_model(tokenizer: &LlmTokenizer, enable_thinking: bool, prompt: &str) -> Self {
        if let Some(ids) = ChannelIds::detect(tokenizer) {
            return Self::Channel(ChannelParser::new(ids));
        }
        if is_llama3_prompt(prompt) {
            return Self::Llama(LlamaJsonParser::new(single_token(tokenizer, "<|python_tag|>")));
        }
        match gemma4_ids(tokenizer) {
            Some(ids) => Self::Gemma(Gemma4StreamParser::new(
                ids,
                prompt.ends_with("<|channel>thought\n"),
            )),
            None => Self::ChatML {
                think: if enable_thinking && knows_think_tokens(tokenizer) {
                    ThinkParser::new_implicit_open()
                } else {
                    ThinkParser::new()
                },
                tools: ToolCallParser::new(),
            },
        }
    }

    pub fn is_channel(&self) -> bool {
        matches!(self, Self::Channel(_))
    }

    pub fn has_native_stops(tokenizer: &LlmTokenizer) -> bool {
        ChannelIds::detect(tokenizer).is_some() || gemma4_ids(tokenizer).is_some()
    }

    pub fn feed(&mut self, id: u32, delta: &str) -> (String, String, String) {
        match self {
            Self::ChatML { think, tools } => {
                let feed = tools.feed(delta);
                let (body, thinking) = if feed.clean_delta.is_empty() {
                    (String::new(), String::new())
                } else {
                    let split = think.feed(&feed.clean_delta);
                    (split.body, split.thinking)
                };
                (body, thinking, feed.tool_delta)
            }
            Self::Channel(p) => {
                let split = p.feed(id, delta);
                (split.body, split.thinking, split.tool)
            }
            Self::Gemma(p) => {
                let split = p.feed(id, delta);
                (split.body, split.thinking, split.tool)
            }
            Self::Llama(p) => {
                let (body, tool) = p.feed(id, delta);
                (body, String::new(), tool)
            }
        }
    }

    pub fn take_tail(&mut self) -> String {
        match self {
            Self::Llama(p) => p.take_tail(),
            _ => String::new(),
        }
    }

    pub fn tool_call_ready(&self) -> bool {
        match self {
            Self::ChatML { tools, .. } => tools.calls_count() > 0 && tools.is_outside(),
            Self::Channel(p) => p.has_closed_call(),
            Self::Gemma(p) => p.has_closed_call(),
            Self::Llama(p) => p.has_closed_call(),
        }
    }

    pub fn finish(self) -> Vec<RawToolCall> {
        match self {
            Self::ChatML { tools, .. } => tools.finish().0,
            Self::Channel(p) => p.finish(),
            Self::Gemma(p) => p
                .finish()
                .into_iter()
                .map(|c| RawToolCall { arguments_json: c.arguments_json(), name: c.name })
                .collect(),
            Self::Llama(p) => p.finish(),
        }
    }
}

fn is_llama3_prompt(prompt: &str) -> bool {
    prompt.trim_end().ends_with("<|start_header_id|>assistant<|end_header_id|>")
}

fn single_token(tokenizer: &LlmTokenizer, s: &str) -> Option<u32> {
    match tokenizer.encode(s) {
        Ok(ids) if ids.len() == 1 => Some(ids[0]),
        _ => None,
    }
}

fn knows_think_tokens(tokenizer: &LlmTokenizer) -> bool {
    matches!(tokenizer.encode("</think>"), Ok(ids) if ids.len() == 1)
}

fn gemma4_ids(tokenizer: &LlmTokenizer) -> Option<Gemma4Ids> {
    Gemma4Ids::detect_with(|s| match tokenizer.encode(s) {
        Ok(ids) if ids.len() == 1 => Some(ids[0]),
        _ => None,
    })
}

pub fn prompt_for_model(tokenizer: &LlmTokenizer, enable_thinking: bool, prompt: String) -> String {
    if enable_thinking && gemma4_ids(tokenizer).is_some() {
        gemma4_restore_empty_thinking(&prompt)
    } else {
        prompt
    }
}

fn gemma4_restore_empty_thinking(prompt: &str) -> String {
    const CLOSE: &str = "<tool_response|>";
    const EMPTY_THOUGHT: &str = "<|channel>thought\n<channel|>";
    let mut out = String::with_capacity(prompt.len() + 64);
    let mut rest = prompt;
    while let Some(pos) = rest.find(CLOSE) {
        let after = pos + CLOSE.len();
        out.push_str(&rest[..after]);
        let tail = &rest[after..];
        if !tail.is_empty() && !tail.starts_with("<|channel>") && !tail.starts_with("<|tool_response>") {
            out.push_str(EMPTY_THOUGHT);
        }
        rest = tail;
    }
    out.push_str(rest);
    out
}


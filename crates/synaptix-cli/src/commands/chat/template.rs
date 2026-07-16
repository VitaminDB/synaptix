use std::path::Path;

use serde_json::Value as Json;
use synaptix_tokenizer::templates::chat_template::RenderOptions;
use synaptix_tokenizer::{ChatTemplate, Message, SpecialTokenKind, SpecialTokens};

use super::app::{Msg, Role};
use super::ChatPipeline;

pub struct Prompt {
    template: Option<ChatTemplate>,
    stop_ids: Vec<u32>,
    enable_thinking: bool,
}

impl Prompt {
    pub fn load(model: &Path, pipeline: &ChatPipeline, enable_thinking: bool) -> Self {
        let specials = pipeline.specials();
        let template = load_template_source(model)
            .map(|src| ChatTemplate::from_source_with_specials(src, specials.clone()));
        let stop_ids = collect_stop_ids(model, pipeline, &specials);
        Self { template, stop_ids, enable_thinking }
    }

    pub fn stop_ids(&self) -> Vec<u32> {
        self.stop_ids.clone()
    }

    pub fn has_template(&self) -> bool {
        self.template.is_some()
    }

    pub fn render(&self, messages: &[Msg], generation_prompt: bool) -> Result<String, String> {
        let msgs = to_messages(messages);
        match &self.template {
            Some(t) => {
                let opts = RenderOptions::new()
                    .with_generation_prompt(generation_prompt)
                    .with_var("enable_thinking", Json::Bool(self.enable_thinking));
                t.render(&msgs, &opts).map_err(|e| e.to_string())
            }
            None => Ok(fallback_render(&msgs, generation_prompt)),
        }
    }
}

fn to_messages(messages: &[Msg]) -> Vec<Message> {
    messages
        .iter()
        .filter(|m| !(m.role == Role::Assistant && m.text.trim().is_empty()))
        .map(|m| match m.role {
            Role::System => Message::system(m.text.clone()),
            Role::User => Message::user(m.text.clone()),
            Role::Assistant => Message::assistant(strip_reasoning(&m.text)),
        })
        .collect()
}

fn strip_reasoning(text: &str) -> String {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim_start().to_string(),
        None => text.to_string(),
    }
}

fn fallback_render(msgs: &[Message], generation_prompt: bool) -> String {
    let mut s = String::new();
    for m in msgs {
        s.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", m.role.as_str(), m.content));
    }
    if generation_prompt {
        s.push_str("<|im_start|>assistant\n");
    }
    s
}

fn load_template_source(model: &Path) -> Option<String> {
    if let Some(bytes) = read_model_file(model, "chat_template.jinja") {
        if let Ok(s) = String::from_utf8(bytes) {
            if !s.trim().is_empty() {
                return Some(s);
            }
        }
    }
    let cfg = read_model_file(model, "tokenizer_config.json")?;
    let v: Json = serde_json::from_slice(&cfg).ok()?;
    v.get("chat_template").and_then(|t| t.as_str()).map(str::to_string)
}

fn collect_stop_ids(model: &Path, pipeline: &ChatPipeline, specials: &SpecialTokens) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let mut push = |id: Option<u32>| {
        if let Some(i) = id {
            if !ids.contains(&i) {
                ids.push(i);
            }
        }
    };
    push(specials.id_of(SpecialTokenKind::ImEnd));
    push(pipeline.token_to_id("<|im_end|>"));
    push(specials.eos_id());
    if let Some(bytes) = read_model_file(model, "generation_config.json") {
        if let Ok(v) = serde_json::from_slice::<Json>(&bytes) {
            match v.get("eos_token_id") {
                Some(Json::Number(n)) => push(n.as_u64().map(|x| x as u32)),
                Some(Json::Array(a)) => {
                    for e in a {
                        push(e.as_u64().map(|x| x as u32));
                    }
                }
                _ => {}
            }
        }
    }
    ids
}

fn read_model_file(model: &Path, name: &str) -> Option<Vec<u8>> {
    if model.is_dir() {
        std::fs::read(model.join(name)).ok()
    } else {
        let bundle = synaptix_bundle::Bundle::open(model).ok()?;
        bundle.read_file(name).ok().map(|c| c.into_owned())
    }
}

use std::collections::BTreeMap;
use std::io::Write;

use serde_json::json;
use synaptix_tokenizer::{
    AddedToken, AddedVocab, ChatTemplate, HfTokenizer, JsonStreamEvent, JsonStreamParser, Message,
    ReasoningEvent, ReasoningStreamParser, SpecialTokenKind, SpecialTokens, Tokenizer,
    ToolCall, ToolCallEvent, ToolCallParser, ToolDef, ToolParamProperty, ToolParameterSchema,
};
use synaptix_tokenizer::templates::chat_template::RenderOptions;

fn write_tokenizer_json(s: &str) -> tempfile::NamedTempFile {
    let mut f = tempfile::Builder::new()
        .suffix(".json")
        .tempfile()
        .expect("tempfile");
    f.write_all(s.as_bytes()).expect("write");
    f
}

#[test]
fn hf_tokenizer_round_trip_minimal_bpe() {
    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": {"a": 0, "b": 1, "ab": 2},
            "merges": [["a", "b"]]
        }
    });
    let f = write_tokenizer_json(&serde_json::to_string(&tokenizer_json).unwrap());
    let tok = HfTokenizer::from_file(f.path()).expect("load");
    assert_eq!(tok.vocab_size(false), 3);
    let enc = tok.encode("ab", false).expect("encode");
    assert_eq!(enc.ids, vec![2]);
    let dec = tok.decode(&[2], false).expect("decode");
    assert_eq!(dec, "ab");
}

#[test]
fn hf_tokenizer_byte_level_bpe_with_specials() {
    let tokenizer_json = json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [
            {"id": 256, "content": "<|im_start|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
            {"id": 257, "content": "<|im_end|>",   "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
        ],
        "normalizer": null,
        "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "byte_fallback": false,
            "ignore_merges": false,
            "vocab": byte_level_vocab(),
            "merges": []
        }
    });
    let f = write_tokenizer_json(&serde_json::to_string(&tokenizer_json).unwrap());
    let tok = HfTokenizer::from_file(f.path()).expect("load");
    assert!(tok.special_tokens().contains(SpecialTokenKind::ImStart));
    assert!(tok.special_tokens().contains(SpecialTokenKind::ImEnd));
    let im_start_id = tok.special_tokens().id_of(SpecialTokenKind::ImStart).unwrap();
    assert_eq!(im_start_id, 256);
    let enc = tok.encode("<|im_start|>hello<|im_end|>", true).expect("encode");
    assert_eq!(enc.ids.first(), Some(&256));
    assert_eq!(enc.ids.last(), Some(&257));
    let dec = tok.decode(&enc.ids, false).expect("decode");
    assert!(dec.contains("<|im_start|>"));
    assert!(dec.contains("hello"));
    assert!(dec.contains("<|im_end|>"));
}

fn byte_level_vocab() -> serde_json::Value {
    let mut m = BTreeMap::new();
    for (i, c) in synaptix_tokenizer::byte_level::BYTE_TO_CHAR.iter().enumerate() {
        m.insert(c.to_string(), i as u32);
    }
    serde_json::to_value(m).unwrap()
}

#[test]
fn added_vocab_split_finds_inserted_token() {
    let mut v = AddedVocab::new();
    v.add(AddedToken::new("<|sep|>", 100).special(true)).unwrap();
    let segments = v.split("hello<|sep|>world");
    assert_eq!(segments.len(), 3);
    match &segments[1] {
        synaptix_tokenizer::added_vocab::Segment::Added { token, .. } => {
            assert_eq!(token.id, 100);
        }
        _ => panic!("expected Added segment"),
    }
}

#[test]
fn chat_template_qwen3_style_simple() {
    let template = r#"{%- for message in messages -%}
{%- if message.role == "system" -%}
<|im_start|>system
{{ message.content }}<|im_end|>
{%- elif message.role == "user" -%}
<|im_start|>user
{{ message.content }}<|im_end|>
{%- elif message.role == "assistant" -%}
<|im_start|>assistant
{{ message.content }}<|im_end|>
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%}"#;
    let mut specials = SpecialTokens::default();
    specials.set(SpecialTokenKind::ImStart, "<|im_start|>", 1);
    specials.set(SpecialTokenKind::ImEnd, "<|im_end|>", 2);
    let tmpl = ChatTemplate::from_source_with_specials(template, specials);
    let msgs = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hi"),
        Message::assistant("Hello!"),
    ];
    let opts = RenderOptions::new().with_generation_prompt(true);
    let rendered = tmpl.render(&msgs, &opts).expect("render");
    assert!(rendered.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
    assert!(rendered.contains("<|im_start|>user\nHi<|im_end|>"));
    assert!(rendered.contains("<|im_start|>assistant\nHello!<|im_end|>"));
    assert!(rendered.ends_with("<|im_start|>assistant"));
}

#[test]
fn chat_template_pycompat_startswith() {
    let template = r#"{%- for m in messages -%}
{%- if m.content.startswith("RUN:") -%}
[CMD]{{ m.content }}[/CMD]
{%- else -%}
{{ m.content }}
{%- endif -%}
{%- endfor -%}"#;
    let tmpl = ChatTemplate::from_source(template);
    let msgs = vec![
        Message::user("RUN: ls -la"),
        Message::user("hello"),
    ];
    let out = tmpl.render(&msgs, &RenderOptions::new()).expect("render");
    assert!(out.contains("[CMD]RUN: ls -la[/CMD]"));
    assert!(out.contains("hello"));
}

#[test]
fn chat_template_renders_tools() {
    let template = r#"{%- for t in tools -%}
TOOL:{{ t.function.name }}
{%- endfor -%}
{%- for m in messages -%}
{{ m.role }}:{{ m.content }};
{%- endfor -%}"#;
    let tools = vec![ToolDef::function(
        "get_weather",
        "Get current weather",
        ToolParameterSchema::object()
            .property("location", ToolParamProperty::new("string").description("city"))
            .require("location"),
    )];
    let tmpl = ChatTemplate::from_source(template);
    let opts = RenderOptions::new().with_tools(tools);
    let out = tmpl
        .render(&[Message::user("What is the weather?")], &opts)
        .expect("render");
    assert!(out.contains("TOOL:get_weather"));
    assert!(out.contains("user:What is the weather?"));
}

#[test]
fn reasoning_then_tool_call_chain() {
    let stream = "<think>Need weather for Tokyo</think><tool_call>{\"name\":\"weather\",\"arguments\":{\"loc\":\"Tokyo\"}}</tool_call>";
    let mut reasoning = ReasoningStreamParser::new();
    let mut tool = ToolCallParser::new();
    let mut all_tool_events = Vec::new();
    let mut thinking = String::new();
    for chunk in stream.as_bytes().chunks(4) {
        for ev in reasoning.push_bytes(chunk) {
            match ev {
                ReasoningEvent::Visible(v) => {
                    all_tool_events.extend(tool.push_str(&v));
                }
                ReasoningEvent::Thinking(t) => thinking.push_str(&t),
            }
        }
    }
    for ev in reasoning.finish() {
        if let ReasoningEvent::Visible(v) = ev {
            all_tool_events.extend(tool.push_str(&v));
        }
    }
    all_tool_events.extend(tool.finish());
    assert_eq!(thinking, "Need weather for Tokyo");
    let calls: Vec<&ToolCallEvent> = all_tool_events
        .iter()
        .filter(|e| matches!(e, ToolCallEvent::ToolCall { .. }))
        .collect();
    assert_eq!(calls.len(), 1);
    match calls[0] {
        ToolCallEvent::ToolCall { name, arguments, .. } => {
            assert_eq!(name, "weather");
            assert!(arguments.contains("Tokyo"));
        }
        _ => unreachable!(),
    }
}

#[test]
fn json_stream_parses_streaming_arguments() {
    let mut p = JsonStreamParser::new();
    let parts = [r#"{"name":""#, r#"weather","args":{"loc":""#, r#"Tokyo"}}"#];
    let mut all = Vec::new();
    for c in parts {
        all.extend(p.push_str(c).unwrap());
    }
    all.extend(p.finish().unwrap());
    assert_eq!(all.len(), 1);
    let JsonStreamEvent::Value(v) = &all[0];
    assert_eq!(v["name"], "weather");
    assert_eq!(v["args"]["loc"], "Tokyo");
}

#[test]
fn tool_call_serialize_roundtrip() {
    let call = ToolCall::function("foo", r#"{"x":1}"#);
    let s = serde_json::to_string(&call).unwrap();
    let back: ToolCall = serde_json::from_str(&s).unwrap();
    assert_eq!(back, call);
}

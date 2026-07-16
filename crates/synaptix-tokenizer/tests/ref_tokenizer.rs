use std::fs;
use std::path::PathBuf;

use serde_json::Value as JsonValue;
use synaptix_test_utils::reference_data_path;
use synaptix_tokenizer::special_tokens::SpecialTokenKind;
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, MessageRole, RenderOptions};
use synaptix_tokenizer::templates::tools::ToolDef;
use synaptix_tokenizer::tokenizer::Tokenizer;
use synaptix_tokenizer::HfTokenizer;

const TOKENIZER_DIR: &str = "models/Qwen/Qwen3-1.7B";

fn load_tokenizer() -> HfTokenizer {
    let path = PathBuf::from(TOKENIZER_DIR).join("tokenizer.json");
    HfTokenizer::from_file(&path)
        .unwrap_or_else(|e| panic!("Не могу загрузить tokenizer.json: {}", e))
}

fn load_chat_template() -> ChatTemplate {
    let cfg_path = PathBuf::from(TOKENIZER_DIR).join("tokenizer_config.json");
    let cfg_bytes = fs::read(&cfg_path)
        .unwrap_or_else(|e| panic!("Не могу прочитать tokenizer_config.json: {}", e));
    let cfg: JsonValue = serde_json::from_slice(&cfg_bytes).unwrap();
    let tmpl = cfg["chat_template"].as_str().expect("chat_template missing").to_string();
    ChatTemplate::from_source(tmpl)
}

fn load_json(name: &str) -> JsonValue {
    let path = reference_data_path("tokenizer", &format!("{}.json", name));
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {:?}: {}", path, e));
    serde_json::from_slice(&bytes).unwrap()
}

fn role_from_str(s: &str) -> MessageRole {
    match s {
        "system" => MessageRole::System,
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "tool" => MessageRole::Tool,
        _ => panic!("unknown role: {}", s),
    }
}

fn messages_from_json(arr: &[JsonValue]) -> Vec<Message> {
    arr.iter()
        .map(|v| {
            let role = role_from_str(v["role"].as_str().unwrap());
            let content = v["content"].as_str().unwrap().to_string();
            Message {
                role,
                content,
                name: None,
                tool_call_id: None,
                tool_calls: Vec::new(),
            }
        })
        .collect()
}

#[test]
fn t10_1_encode_decode() {
    let tok = load_tokenizer();
    let cases = load_json("encode_decode");
    for case in cases.as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected_ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected_decoded = case["decoded"].as_str().unwrap();
        let enc = tok.encode(text, false).unwrap();
        assert_eq!(enc.ids, expected_ids, "ids mismatch for text: {:?}", text);
        let decoded = tok.decode(&enc.ids, false).unwrap();
        assert_eq!(decoded, expected_decoded, "decoded mismatch for text: {:?}", text);
    }
}

#[test]
fn t10_2_batch_encode() {
    let tok = load_tokenizer();
    let data = load_json("batch_encode");
    let texts: Vec<String> = data["texts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let expected_ids: Vec<Vec<u32>> = data["input_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect()
        })
        .collect();
    let encs = tok.encode_batch(&texts, true).unwrap();
    let max_len = expected_ids.iter().map(|v| v.len()).max().unwrap();
    let pad_id = expected_ids
        .iter()
        .flat_map(|v| v.iter())
        .find(|&&id| {
            expected_ids.iter().any(|row| row.last() == Some(&id) && row.len() < max_len)
        })
        .copied()
        .unwrap_or(151643);
    for (i, enc) in encs.iter().enumerate() {
        let mut padded = enc.ids.clone();
        while padded.len() < max_len {
            padded.push(pad_id);
        }
        assert_eq!(padded, expected_ids[i], "batch[{}] ids mismatch (padded)", i);
    }
}

#[test]
fn t10_3_chat_template() {
    let tmpl = load_chat_template();
    let data = load_json("chat_template");
    let messages = messages_from_json(data["messages"].as_array().unwrap());
    let opts = RenderOptions::new().with_generation_prompt(true);
    let result = tmpl
        .render(&messages, &opts)
        .unwrap_or_else(|e| panic!("render failed: {:?}", e));
    let expected = data["formatted"].as_str().unwrap();
    assert_eq!(result, expected);
}

#[test]
fn t10_4_tools_template() {
    let tmpl = load_chat_template();
    let data = load_json("tools_template");
    let messages = messages_from_json(data["messages"].as_array().unwrap());
    let tools_json = data["tools"].as_array().unwrap();
    let tools: Vec<ToolDef> = tools_json
        .iter()
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .collect();
    let opts = RenderOptions::new().with_generation_prompt(true).with_tools(tools);
    let result = tmpl
        .render(&messages, &opts)
        .unwrap_or_else(|e| panic!("render failed: {:?}", e));
    let expected = data["formatted"].as_str().unwrap();
    assert_eq!(result, expected);
}

#[test]
fn t10_5_special_tokens() {
    let tok = load_tokenizer();
    let data = load_json("special_tokens");
    let eos = data["eos_token"].as_str().unwrap();
    let eos_id = data["eos_token_id"].as_u64().unwrap() as u32;
    assert_eq!(tok.token_to_id(eos), Some(eos_id), "eos_token_id mismatch");
    if let Some(pad) = data["pad_token"].as_str() {
        let pad_id = data["pad_token_id"].as_u64().unwrap() as u32;
        assert_eq!(tok.token_to_id(pad), Some(pad_id), "pad_token_id mismatch");
    }
    let _ = SpecialTokenKind::Eos;
}


#[test]
fn t10_6_long_text() {
    let tok = load_tokenizer();
    let data = load_json("long_text");
    let total_len = data["len"].as_u64().unwrap() as usize;
    let first_32: Vec<u32> = data["first_32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let last_32: Vec<u32> = data["last_32"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let text = "The transformer architecture revolutionized natural language processing. ".repeat(128);
    let enc = tok.encode(&text, false).unwrap();
    assert_eq!(enc.ids.len(), total_len, "длина не совпадает");
    assert_eq!(&enc.ids[..32], &first_32[..], "первые 32 ids");
    let tail_off = enc.ids.len() - 32;
    assert_eq!(&enc.ids[tail_off..], &last_32[..], "последние 32 ids");
}

#[test]
fn t10_7_unicode_edge() {
    let tok = load_tokenizer();
    let cases = load_json("unicode_edge");
    for case in cases.as_array().unwrap() {
        let text = case["text"].as_str().unwrap();
        let expected_ids: Vec<u32> = case["ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let expected_decoded = case["decoded"].as_str().unwrap();
        let enc = tok.encode(text, false).unwrap();
        assert_eq!(enc.ids, expected_ids, "unicode encode ids mismatch: {:?}", text);
        let decoded = tok.decode(&enc.ids, false).unwrap();
        assert_eq!(decoded, expected_decoded, "unicode decode mismatch: {:?}", text);
    }
}

#[test]
fn t10_8_streaming_detok() {
    let tok = load_tokenizer();
    let data = load_json("streaming_detok");
    let steps = data["steps"].as_array().unwrap();
    for (i, step) in steps.iter().enumerate() {
        let id = step["id"].as_u64().unwrap() as u32;
        let expected_piece = step["piece"].as_str().unwrap();
        let piece = tok.decode(&[id], true).unwrap();
        assert_eq!(piece, expected_piece, "step[{}] piece mismatch (id={})", i, id);
    }
}

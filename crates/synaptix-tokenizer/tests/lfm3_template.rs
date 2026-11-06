//! Рендер chat_template.jinja LFM3 (тег `{% generation %}` transformers) — байт в байт как
//! `apply_chat_template` transformers 4.57.

use serde_json::json;
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, RenderOptions};

const TEMPLATE: &str = include_str!("fixtures/lfm3_chat_template.jinja");

fn msgs() -> Vec<Message> {
    vec![Message::system("Ты помощник."), Message::user("Привет"), Message::assistant("Здравствуйте!"), Message::user("Как дела?")]
}

#[test]
fn lfm3_like_transformers() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let base = RenderOptions::new().with_generation_prompt(true).with_var("bos_token", json!("<|startoftext|>"));
    let out = t.render(&msgs(), &base).expect("render");
    assert_eq!(out, "<|startoftext|><|im_start|>system\nТы помощник.<|im_end|>\n<|im_start|>user\nПривет<|im_end|>\n<|im_start|>assistant\n<think></think>Здравствуйте!<|im_end|>\n<|im_start|>user\nКак дела?<|im_end|>\n<|im_start|>assistant\n<think>");
    let out = t.render(&msgs(), &base.clone().with_var("enable_thinking", json!(false))).expect("render");
    assert!(out.ends_with("<|im_start|>assistant\n<think></think>"), "{out}");
}

use serde_json::json;
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, RenderOptions};

const TEMPLATE: &str = include_str!("fixtures/muse_glimmer_chat_template.jinja");

fn msgs() -> Vec<Message> {
    vec![
        Message::system("You are a helpful assistant."),
        Message::user("Столица Франции?"),
    ]
}

#[test]
fn renders_channel_protocol() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(&msgs(), &RenderOptions::new().with_generation_prompt(true))
        .expect("render default");
    assert!(out.contains("<|start|>system<|message|>You are a helpful assistant."), "{out}");
    assert!(out.contains("Reasoning strength: high."), "{out}");
    assert!(out.contains("# Valid recipients: \"self\", \"user\"."), "{out}");
    assert!(out.contains("<|start|>user<|message|>Столица Франции?<|eot|>"), "{out}");
    assert!(out.ends_with("<|start|>assistant"), "{out}");
}

#[test]
fn default_system_block_when_absent() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(
            &[Message::user("hi")],
            &RenderOptions::new().with_generation_prompt(true),
        )
        .expect("render no-system");
    assert!(out.contains("You are a helpful AI assistant."), "{out}");
    assert!(out.contains("Knowledge cutoff: 2026-01-04."), "{out}");
    assert!(out.contains("Reasoning strength: high."), "{out}");
}

#[test]
fn reasoning_strength_var_overrides() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(
            &msgs(),
            &RenderOptions::new()
                .with_generation_prompt(true)
                .with_var("reasoning_strength", json!("low")),
        )
        .expect("render low");
    assert!(out.contains("Reasoning strength: low."), "{out}");
}

#[test]
fn reasoning_effort_in_system_prompt_is_normalized() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(
            &[
                Message::system("Be terse. Reasoning effort: medium."),
                Message::user("hi"),
            ],
            &RenderOptions::new().with_generation_prompt(true),
        )
        .expect("render normalized");
    assert!(out.contains("Reasoning strength: medium."), "{out}");
    assert!(!out.contains("Reasoning strength: high."), "{out}");
}

#[test]
fn assistant_reply_ends_with_eot() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(
            &[
                Message::user("2+2?"),
                Message::assistant("4"),
                Message::user("а 3+3?"),
            ],
            &RenderOptions::new().with_generation_prompt(true),
        )
        .expect("render multi-turn");
    assert!(out.contains("<|start|>assistant to=user<|message|>4<|eot|>"), "{out}");
    assert!(out.ends_with("<|start|>assistant"), "{out}");
}

//! Рендер chat_template.jinja Qwen3.8-27B (transformers 5.x): новый блок
//! reasoning_effort (xhigh/medium/low + raise_exception на мусор), preserved
//! thinking по умолчанию и совместимость с прежним qwen3_5-форматом im_start.

use serde_json::json;
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, RenderOptions};

const TEMPLATE: &str = include_str!("fixtures/qwen3_8_chat_template.jinja");

fn msgs() -> Vec<Message> {
    vec![
        Message::system("You are a helpful assistant."),
        Message::user("Столица Франции?"),
    ]
}

#[test]
fn renders_default_with_xhigh_reasoning() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(&msgs(), &RenderOptions::new().with_generation_prompt(true))
        .expect("render default");
    assert!(out.contains("<|im_start|>system\n"), "{out}");
    assert!(
        out.contains("Reasoning effort is set to xhigh."),
        "дефолт reasoning_effort должен быть xhigh: {out}"
    );
    assert!(out.contains("You are a helpful assistant."), "{out}");
    assert!(out.contains("<|im_start|>user\nСтолица Франции?<|im_end|>\n"), "{out}");
    assert!(out.ends_with("<|im_start|>assistant\n<think>\n"), "{out}");
}

#[test]
fn reasoning_effort_low_and_medium() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let low = t
        .render(
            &msgs(),
            &RenderOptions::new()
                .with_generation_prompt(true)
                .with_var("reasoning_effort", json!("low")),
        )
        .expect("render low");
    assert!(low.contains("Reasoning effort is set to low."), "{low}");

    // medium — без инструкции вовсе (в шаблоне нет ветки для medium).
    let med = t
        .render(
            &msgs(),
            &RenderOptions::new()
                .with_generation_prompt(true)
                .with_var("reasoning_effort", json!("medium")),
        )
        .expect("render medium");
    assert!(!med.contains("Reasoning effort"), "{med}");
}

#[test]
fn invalid_reasoning_effort_raises() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let err = t.render(
        &msgs(),
        &RenderOptions::new()
            .with_generation_prompt(true)
            .with_var("reasoning_effort", json!("turbo")),
    );
    assert!(err.is_err(), "мусорный reasoning_effort обязан падать raise_exception");
}

#[test]
fn enable_thinking_false_forces_empty_think() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let out = t
        .render(
            &msgs(),
            &RenderOptions::new()
                .with_generation_prompt(true)
                .with_var("enable_thinking", json!(false)),
        )
        .expect("render no-think");
    assert!(!out.contains("Reasoning effort"), "{out}");
    assert!(out.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{out}");
}

#[test]
fn multiturn_preserves_thinking_by_default() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let history = vec![
        Message::user("2+2?"),
        Message::assistant("<think>\nПросто арифметика.\n</think>\n\n4"),
        Message::user("А 3+3?"),
    ];
    let out = t
        .render(&history, &RenderOptions::new().with_generation_prompt(true))
        .expect("render multiturn");
    // Qwen3.8: preserve_thinking undefined → true, think-блок истории сохраняется.
    assert!(out.contains("Просто арифметика."), "{out}");
}

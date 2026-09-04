//! Рендер chat_template.jinja Qwen3.8-27B (transformers 5.x): новый блок
//! reasoning_effort (xhigh/medium/low + raise_exception на мусор), preserved
//! thinking по умолчанию и совместимость с прежним qwen3_5-форматом im_start.

use serde_json::{json, Value as JsonValue};
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, RenderOptions};
use synaptix_tokenizer::templates::tools::{ToolCall, ToolCallFunction};

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


fn tools_var() -> JsonValue {
    json!([{
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"]
            }
        }
    }])
}

fn call(name: &str, arguments: &str) -> ToolCall {
    ToolCall {
        id: None,
        call_type: "function".into(),
        function: ToolCallFunction { name: name.into(), arguments: arguments.into() },
    }
}

/// Инвариант префикс-KV агента: промпт хода (`add_generation_prompt`) обязан
/// оставаться префиксом промпта следующего хода. Реплика ассистента для
/// этого несёт и размышления, и вызовы: промпт хода кончается
/// `…assistant\n<think>\n`, и без `reasoning_content` шаблон рендерит
/// `<think>\n\n</think>` — префикс рвётся на первом же ходу с инструментом
/// (разбор 04.09.2026, flash-next терял весь кэш каждый ход).
#[test]
fn assistant_turn_with_reasoning_keeps_previous_prompt_as_prefix() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let opts = || {
        RenderOptions::new()
            .with_generation_prompt(true)
            .with_var("tools", tools_var())
    };

    let base = msgs();
    let turn_prompt = t.render(&base, &opts()).expect("render turn");
    assert!(turn_prompt.ends_with("<|im_start|>assistant\n<think>\n"), "{turn_prompt}");

    let mut next = base.clone();
    next.push(Message::assistant_with_reasoning(
        "",
        Some("Нужно спросить систему.".into()),
        vec![call("bash", r#"{"command":"uname -r"}"#)],
    ));
    next.push(Message::tool("7.0.10", "call-1"));
    let next_prompt = t.render(&next, &opts()).expect("render next");

    assert!(
        next_prompt.starts_with(&turn_prompt),
        "промпт следующего хода обязан продолжать прошлый.\nбыло:\n{turn_prompt}\nстало:\n{next_prompt}"
    );
    // Вызов ушёл в родном формате шаблона, а не текстом в content.
    assert!(next_prompt.contains("<tool_call>\n<function=bash>\n"), "{next_prompt}");
    assert!(
        next_prompt.contains("<parameter=command>\nuname -r\n</parameter>"),
        "аргументы должны развернуться в параметры: {next_prompt}"
    );
    assert!(next_prompt.contains("Нужно спросить систему."), "{next_prompt}");
}

/// Без размышлений (`enable_thinking = false`) промпт хода кончается на
/// `<think>\n\n</think>\n\n`, и реплика без reasoning его продолжает.
#[test]
fn assistant_turn_without_thinking_keeps_prefix_too() {
    let t = ChatTemplate::from_source(TEMPLATE);
    let opts = || {
        RenderOptions::new()
            .with_generation_prompt(true)
            .with_var("enable_thinking", json!(false))
            .with_var("tools", tools_var())
    };
    let base = msgs();
    let turn_prompt = t.render(&base, &opts()).expect("render turn");

    let mut next = base.clone();
    next.push(Message::assistant_with_reasoning(
        "",
        None,
        vec![call("bash", r#"{"command":"pwd"}"#)],
    ));
    next.push(Message::tool("/tmp", "call-1"));
    let next_prompt = t.render(&next, &opts()).expect("render next");
    assert!(
        next_prompt.starts_with(&turn_prompt),
        "было:\n{turn_prompt}\nстало:\n{next_prompt}"
    );
}

//! Рендер chat_template.jinja Inkling-2-Mini байт в байт как transformers 4.57
//! (`_compile_jinja_template` → `render`): `namespace`, `{% set %}…{% endset %}`,
//! макрос с `reasoning_effort` (строки уровней и число), `raise_exception`,
//! `tojson(sort_keys=true, separators=(",", ":"))`, `dictsort(case_sensitive=true)`
//! и тег `{% generation %}`. Шаблон без приглашения к ответу: промпт кончается
//! репликой пользователя, ответ модель начинает сама с `<|message_model|>`.

use serde_json::json;
use synaptix_tokenizer::templates::chat_template::{ChatTemplate, Message, RenderOptions};
use synaptix_tokenizer::templates::tools::{ToolCall, ToolDef};

const TEMPLATE: &str = include_str!("fixtures/inkling2_chat_template.jinja");

fn render(msgs: &[Message], opts: RenderOptions) -> String {
    ChatTemplate::from_source(TEMPLATE)
        .render(msgs, &opts.with_generation_prompt(true))
        .expect("render")
}

fn effort(level: serde_json::Value) -> String {
    render(&[Message::user("hi")], RenderOptions::new().with_var("reasoning_effort", level))
}

#[test]
fn system_and_default_effort() {
    let out = render(&[Message::system("Ты помощник."), Message::user("Привет")], RenderOptions::new());
    assert_eq!(
        out,
        "<|message_system|><|content_text|>Ты помощник.<|end_message|><|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Привет<|end_message|>"
    );
}

#[test]
fn effort_levels_and_number() {
    let head = |v: &str| {
        format!("<|message_system|><|content_text|>Thinking effort level: {v}<|end_message|><|message_user|><|content_text|>hi<|end_message|>")
    };
    assert_eq!(effort(json!("low")), head("0.2"));
    assert_eq!(effort(json!("none")), head("0"));
    assert_eq!(effort(json!("max")), head("0.99"));
    assert_eq!(effort(json!("xhigh")), head("0.99"));
    assert_eq!(effort(json!(0.5)), head("0.5"));
}

#[test]
fn unknown_effort_raises() {
    let err = ChatTemplate::from_source(TEMPLATE)
        .render(&[Message::user("hi")], &RenderOptions::new().with_var("reasoning_effort", json!("turbo")))
        .unwrap_err();
    assert!(err.to_string().contains("Unknown reasoning_effort: turbo"), "{err}");
}

#[test]
fn multi_turn_with_reasoning() {
    let out = render(
        &[
            Message::user("2+2?"),
            Message::assistant_with_reasoning("4", Some("считаю".into()), Vec::new()),
            Message::user("а 3+3?"),
        ],
        RenderOptions::new(),
    );
    assert_eq!(
        out,
        "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>2+2?<|end_message|><|message_model|><|content_thinking|>считаю<|end_message|><|message_model|><|content_text|>4<|end_message|><|content_model_end_sampling|><|message_user|><|content_text|>а 3+3?<|end_message|>"
    );
}

#[test]
fn tools_declaration_and_calls() {
    let tool: ToolDef = serde_json::from_value(json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Погода в городе",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Город"},
                    "units": {"type": "string", "enum": ["c", "f"]}
                },
                "required": ["city"]
            }
        }
    }))
    .unwrap();
    let mut call = ToolCall::function("get_weather", r#"{"units":"c","city":"Алматы","opts":{"z":1,"a":[1,2]}}"#);
    call.id = Some("c1".into());
    let out = render(
        &[
            Message::user("Погода в Алматы?"),
            Message::assistant_with_tools("", vec![call]),
            Message::tool("+5", "c1"),
        ],
        RenderOptions::new().with_tools(vec![tool]),
    );
    assert_eq!(
        out,
        concat!(
            r#"<|message_system|>tool_declare<|content_xml|>[{"description":"Погода в городе","name":"get_weather","parameters":{"properties":{"city":{"description":"Город","type":"string"},"units":{"enum":["c","f"],"type":"string"}},"required":["city"],"type":"object"},"type":"function"}]<|end_message|>"#,
            "<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|><|content_text|>Погода в Алматы?<|end_message|>",
            r#"<|message_model|>get_weather<|content_invoke_tool_json|>{"name":"get_weather","args":{"city":"Алматы","opts":{"z":1,"a":[1,2]},"units":"c"}}<|end_message|><|content_model_end_sampling|>"#,
            "<|message_tool|>get_weather<|content_text|>+5<|end_message|>"
        )
    );
}

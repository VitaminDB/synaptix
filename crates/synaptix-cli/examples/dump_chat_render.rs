//! Дамп РЕАЛЬНОГО chat-рендера (jinja-шаблон модели) на multi-turn диалоге —
//! чтобы увидеть, что именно чат шлёт в модель (теряется ли история).
//! cargo run --profile fast-release -p synaptix-cli --example dump_chat_render -- MODEL.syn
use synaptix_tokenizer::{ChatTemplate, Message, SpecialTokens};
use synaptix_tokenizer::templates::chat_template::RenderOptions;
use serde_json::Value as Json;

fn main() {
    let path = std::env::args().nth(1).expect("usage: dump_chat_render MODEL.syn");
    let p = std::path::Path::new(&path);
    let bundle = synaptix_bundle::Bundle::open(p).expect("open bundle");
    // источник шаблона: chat_template.jinja либо tokenizer_config.chat_template
    let src = bundle.read_file("chat_template.jinja").ok().map(|c| String::from_utf8_lossy(&c).into_owned())
        .filter(|s| !s.trim().is_empty())
        .or_else(|| {
            let cfg = bundle.read_file("tokenizer_config.json").ok()?;
            let v: Json = serde_json::from_slice(&cfg).ok()?;
            v.get("chat_template").and_then(|t| t.as_str()).map(str::to_string)
        })
        .expect("no chat_template");
    eprintln!("=== ШАБЛОН найден, длина {} символов ===", src.len());

    let tmpl = ChatTemplate::from_source_with_specials(src, SpecialTokens::default());
    let msgs = vec![
        Message::user("Запомни: меня зовут Алексей Петров, инженер из Казани."),
        Message::assistant("Привет, Алексей! Запомнил."),
        Message::user("Расскажи про Париж."),
        Message::assistant("Париж — столица Франции."),
        Message::user("Как меня зовут?"),
    ];
    for think in [true, false] {
        let opts = RenderOptions::new()
            .with_generation_prompt(true)
            .with_var("enable_thinking", Json::Bool(think));
        match tmpl.render(&msgs, &opts) {
            Ok(s) => {
                println!("\n========== РЕНДЕР (enable_thinking={think}) {} символов ==========", s.len());
                println!("{s}");
                println!("--- содержит 'Алексей': {} | 'Казани': {} | 5 ролей user/assistant ---",
                    s.contains("Алексей"), s.contains("Казани"));
            }
            Err(e) => println!("RENDER ERROR (think={think}): {e}"),
        }
    }
}

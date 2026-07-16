use std::path::PathBuf;

use synaptix_tokenizer::templates::chat_template::RenderOptions;
use synaptix_tokenizer::{ChatTemplate, HfTokenizer, Message, SpecialTokenKind, Tokenizer};

fn bundle() -> Option<PathBuf> {
    let p = PathBuf::from("models/qwen3.6 27B.syn");
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

#[test]
fn qwen36_chat_template_renders_and_stops() {
    let Some(path) = bundle() else { return };
    let b = synaptix_bundle::Bundle::open(&path).expect("open bundle");
    let src = String::from_utf8(
        b.read_file("chat_template.jinja").expect("chat_template.jinja").into_owned(),
    )
    .expect("utf8");
    let tok_json = b.read_file("tokenizer.json").expect("tokenizer.json").into_owned();
    let tok = HfTokenizer::from_bytes(&tok_json).expect("tokenizer");
    let specials = tok.special_tokens().clone();

    let im_end = specials
        .id_of(SpecialTokenKind::ImEnd)
        .or_else(|| tok.token_to_id("<|im_end|>"));
    eprintln!(
        "[stop ids] im_end={:?} eos={:?} endoftext={:?}",
        im_end,
        specials.eos_id(),
        tok.token_to_id("<|endoftext|>")
    );
    assert!(im_end.is_some(), "<|im_end|> id не найден — chat не сможет остановиться");

    let numbered: String = src
        .lines()
        .enumerate()
        .filter(|(i, _)| (88..=100).contains(&(i + 1)))
        .map(|(i, l)| format!("{:>3}: {l}\n", i + 1))
        .collect();
    eprintln!("--- template lines 88..100 ---\n{numbered}");

    let tmpl = ChatTemplate::from_source_with_specials(src, specials);

    // Контент ассистента из реального хода: открывающего <think> НЕТ (он был в
    // промпте), есть только </think> в середине → strip_reasoning должен оставить
    // только ответ, иначе template на строке 95 падает ("too many arguments").
    let raw_assistant = "Here's a thinking process:\n1. greet\n</think>\n\nПривет! Чем могу помочь?";
    let multi = vec![
        Message::user("Привет"),
        Message::assistant(strip_reasoning(raw_assistant)),
        Message::user("Расскажи что нового"),
    ];
    let single = vec![Message::user("Привет, как дела?")];

    for (name, msgs) in [("single", &single), ("multi", &multi)] {
        let opts = RenderOptions::new()
            .with_generation_prompt(true)
            .with_var("enable_thinking", serde_json::Value::Bool(true));
        match tmpl.render(msgs, &opts) {
            Ok(rendered) => {
                eprintln!("--- {name} OK ---\n{rendered}\n");
                assert!(rendered.contains("<|im_start|>assistant"));
            }
            Err(e) => panic!("[{name}] render FAILED: {e}"),
        }
    }
}

fn strip_reasoning(text: &str) -> String {
    match text.rfind("</think>") {
        Some(i) => text[i + "</think>".len()..].trim_start().to_string(),
        None => text.to_string(),
    }
}

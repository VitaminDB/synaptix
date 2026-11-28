//! Граница головы промпта (`prefix_messages` службы) на токенах модели:
//! `chat_prefix_check <модель.syn> [N]` — для чата [system, user(бриф A)] и
//! [system, user(бриф B)] голова из первых N сообщений должна лечь на токены,
//! совпасть у обоих промптов и быть их общим префиксом. Без видеокарты.
use synaptix::facade::chat::{head_tokens, prompt_for_model};
use synaptix::facade::llm::{load_llm_tokenizer, Message};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let a: Vec<String> = std::env::args().collect();
    let tok = load_llm_tokenizer(std::path::Path::new(&a[1]))?;
    let n: usize = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(1);
    let head = "Ты — сценарист съёмочной группы. Правила роли: … ".repeat(200);
    let tools = vec![serde_json::json!({"type":"function","function":{"name":"write_note","description":"Записка другой роли","parameters":{"type":"object","properties":{"to":{"type":"string"},"text":{"type":"string"}},"required":["to","text"]}}})];
    let mut ok = true;
    for thinking in [false, true] {
        for with_tools in [false, true] {
            let t = with_tools.then_some(&tools[..]);
            let mut heads = Vec::new();
            for brief in ["Бриф сцены 3: перевести диалог.", "Бриф сцены 7: сверить имена героев и реквизит."] {
                let msgs = vec![Message::system(head.clone()), Message::user(brief.to_string())];
                let prompt = prompt_for_model(&tok, thinking, tok.apply_chat_template_reasoning(&msgs, true, thinking, None, t)?);
                let ids = tok.encode(&prompt)?;
                let h = head_tokens(&tok, &msgs, n, thinking, None, t, &prompt, &ids);
                println!("thinking={thinking} tools={with_tools} промпт {} токенов, голова {h:?}", ids.len());
                match h {
                    Some(h) => heads.push(ids[..h].to_vec()),
                    None => ok = false,
                }
            }
            if heads.len() == 2 && heads[0] != heads[1] {
                println!("  головы РАЗНЫЕ");
                ok = false;
            }
        }
    }
    println!("{}", if ok { "ok: голова ложится на токены и общая у разных брифов" } else { "ОШИБКА" });
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}

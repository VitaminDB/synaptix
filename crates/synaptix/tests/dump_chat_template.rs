use std::path::Path;

const MODEL: &str = "/home/master/models/Qwen3.6-27B-Fable-Fusion-711-MTP.syn";

#[test]
#[ignore]
fn dump_chat_template() {
    let p = Path::new(MODEL);
    let jinja = synaptix::facade::arch::read_model_file(p, "chat_template.jinja");
    let tokcfg = synaptix::facade::arch::read_model_file(p, "tokenizer_config.json");
    println!(
        "chat_template.jinja: {:?} байт",
        jinja.as_ref().map(|b| b.len())
    );
    println!(
        "tokenizer_config.json: {:?} байт",
        tokcfg.as_ref().map(|b| b.len())
    );

    let src = jinja
        .and_then(|b| String::from_utf8(b).ok())
        .or_else(|| {
            let cfg = tokcfg?;
            let v: serde_json::Value = serde_json::from_slice(&cfg).ok()?;
            v.get("chat_template")
                .and_then(|t| t.as_str())
                .map(str::to_string)
        })
        .expect("шаблон не найден");

    println!("длина шаблона = {}", src.len());
    for (i, line) in src.lines().enumerate() {
        if line.contains("tool_call") || line.contains("tools") || line.contains("function") {
            println!("{i:4}: {line}");
        }
    }
}

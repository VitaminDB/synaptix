use std::fmt::Write as FmtWrite;
use super::{StreamingDelta, StreamingFinal};

pub fn delta_to_sse(delta: &StreamingDelta) -> String {
    let mut s = String::new();
    let json = delta_to_json(delta);
    let _ = writeln!(s, "data: {json}");
    let _ = writeln!(s);
    s
}

pub fn final_to_sse(fin: &StreamingFinal) -> String {
    let mut s = String::new();
    let json = final_to_json(fin);
    let _ = writeln!(s, "data: {json}");
    let _ = writeln!(s);
    s
}

pub fn done_event() -> &'static str {
    "data: [DONE]\n\n"
}

fn delta_to_json(d: &StreamingDelta) -> String {
    let text = d.text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    let finish = match &d.finish_reason {
        None => "null".into(),
        Some(r) => format!("\"{}\"", format!("{r:?}").to_lowercase()),
    };
    format!(
        r#"{{"id":{},"index":{},"text":"{}","finish_reason":{}}}"#,
        d.request_id, d.index, text, finish
    )
}

fn final_to_json(f: &StreamingFinal) -> String {
    let text = f.generated_text.replace('\\', "\\\\").replace('"', "\\\"").replace('\n', "\\n");
    format!(
        r#"{{"id":{},"text":"{}","stop_reason":"{}","prompt_tokens":{},"generated_tokens":{}}}"#,
        f.request_id, text,
        format!("{:?}", f.stop_reason).to_lowercase(),
        f.num_prompt_tokens, f.num_generated_tokens
    )
}

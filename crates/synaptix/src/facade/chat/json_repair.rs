pub fn repair_bracket_tail(text: &str) -> Option<String> {
    let head = text.trim_end_matches(|c: char| c == '}' || c == ']' || c.is_whitespace());
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for c in head.chars() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                if stack.pop()? != c {
                    return None;
                }
            }
            _ => {}
        }
    }
    if in_string || stack.is_empty() {
        return None;
    }
    let mut out = head.to_string();
    while let Some(c) = stack.pop() {
        out.push(c);
    }
    (out != text.trim_end()).then_some(out)
}

pub fn parse_with_repair(text: &str) -> Option<(serde_json::Value, bool)> {
    if let Ok(v) = serde_json::from_str(text) {
        return Some((v, false));
    }
    let fixed = repair_bracket_tail(text)?;
    serde_json::from_str(&fixed).ok().map(|v| (v, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repairs_out_of_order_tail_from_real_tool_call() {
        let broken = r#"{"name":"pipelines","arguments":{"action":"apply","set_state":[{"node":9,"state":{"kind":"TextView","data":{"output_text":"[verse]"}}]}}"#;
        let (v, repaired) = parse_with_repair(broken).expect("чинится");
        assert!(repaired);
        assert_eq!(v["name"], "pipelines");
        assert_eq!(v["arguments"]["set_state"][0]["node"], 9);
        assert_eq!(
            v["arguments"]["set_state"][0]["state"]["data"]["output_text"],
            "[verse]"
        );
    }

    #[test]
    fn closes_missing_brackets() {
        assert_eq!(repair_bracket_tail(r#"[{"a": 1}"#).as_deref(), Some(r#"[{"a": 1}]"#));
    }

    #[test]
    fn leaves_intact_and_gives_up_on_truncated_string() {
        assert!(repair_bracket_tail(r#"[{"a": 1}]"#).is_none());
        assert!(repair_bracket_tail(r#"[{"a": "unterminated"#).is_none());
        assert!(repair_bracket_tail(r#"{"a": [1}, "b": 2}"#).is_none());
        assert_eq!(repair_bracket_tail(r#"[{"a": 1]"#).as_deref(), Some(r#"[{"a": 1}]"#));
        let (_, repaired) = parse_with_repair(r#"{"a": 1}"#).expect("целый JSON");
        assert!(!repaired);
    }
}

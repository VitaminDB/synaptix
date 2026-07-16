use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolCallEvent {
    Text(String),
    ToolCall {
        id: Option<String>,
        name: String,
        arguments: String,
        raw: String,
    },
}

pub struct ToolCallParser {
    open_tag: String,
    close_tag: String,
    state: State,
    buffer: String,
    pending_bytes: Vec<u8>,
}

#[derive(Debug)]
enum State {
    Text,
    InToolCall,
}

impl Default for ToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ToolCallParser {
    pub fn new() -> Self {
        Self::with_tags("<tool_call>", "</tool_call>")
    }

    pub fn with_tags(open: impl Into<String>, close: impl Into<String>) -> Self {
        Self {
            open_tag: open.into(),
            close_tag: close.into(),
            state: State::Text,
            buffer: String::new(),
            pending_bytes: Vec::new(),
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Vec<ToolCallEvent> {
        self.pending_bytes.extend_from_slice(bytes);
        let valid_up_to = match std::str::from_utf8(&self.pending_bytes) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_up_to == 0 {
            return Vec::new();
        }
        let valid = std::str::from_utf8(&self.pending_bytes[..valid_up_to])
            .expect("valid_up_to slice must be utf-8")
            .to_owned();
        self.pending_bytes.drain(..valid_up_to);
        self.push_str(&valid)
    }

    pub fn push_str(&mut self, s: &str) -> Vec<ToolCallEvent> {
        self.buffer.push_str(s);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Text => {
                    if let Some(pos) = self.buffer.find(&self.open_tag) {
                        if pos > 0 {
                            out.push(ToolCallEvent::Text(self.buffer[..pos].to_string()));
                        }
                        let after = pos + self.open_tag.len();
                        self.buffer.drain(..after);
                        self.state = State::InToolCall;
                        continue;
                    }
                    let safe = safe_emit_len(&self.buffer, &self.open_tag);
                    if safe > 0 {
                        out.push(ToolCallEvent::Text(self.buffer[..safe].to_string()));
                        self.buffer.drain(..safe);
                    }
                    break;
                }
                State::InToolCall => {
                    if let Some(pos) = self.buffer.find(&self.close_tag) {
                        let raw = self.buffer[..pos].to_string();
                        let after = pos + self.close_tag.len();
                        self.buffer.drain(..after);
                        out.push(parse_tool_call(&raw));
                        self.state = State::Text;
                        continue;
                    }
                    break;
                }
            }
        }
        out
    }

    pub fn finish(mut self) -> Vec<ToolCallEvent> {
        let mut out = Vec::new();
        match self.state {
            State::Text => {
                if !self.buffer.is_empty() {
                    out.push(ToolCallEvent::Text(std::mem::take(&mut self.buffer)));
                }
            }
            State::InToolCall => {
                if !self.buffer.is_empty() {
                    let leftover = std::mem::take(&mut self.buffer);
                    out.push(ToolCallEvent::Text(format!("{}{}", self.open_tag, leftover)));
                }
            }
        }
        if !self.pending_bytes.is_empty() {
            out.push(ToolCallEvent::Text(
                String::from_utf8_lossy(&self.pending_bytes).into_owned(),
            ));
        }
        out
    }
}

fn safe_emit_len(buf: &str, tag: &str) -> usize {
    let buf_bytes = buf.as_bytes();
    let tag_bytes = tag.as_bytes();
    let max = tag_bytes.len().min(buf_bytes.len());
    let mut max_k = 0;
    for k in 1..=max {
        if &buf_bytes[buf_bytes.len() - k..] == &tag_bytes[..k] {
            max_k = k;
        }
    }
    let cut = buf.len().saturating_sub(max_k);
    let mut cut = cut;
    while cut > 0 && !buf.is_char_boundary(cut) {
        cut -= 1;
    }
    cut
}

fn parse_tool_call(raw: &str) -> ToolCallEvent {
    let trimmed = raw.trim();
    if let Ok(JsonValue::Object(obj)) = serde_json::from_str::<JsonValue>(trimmed) {
        let id = obj.get("id").and_then(|v| v.as_str()).map(String::from);
        let name = obj
            .get("name")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_default();
        let arguments = match obj.get("arguments") {
            Some(JsonValue::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".into(),
        };
        ToolCallEvent::ToolCall { id, name, arguments, raw: raw.to_string() }
    } else {
        ToolCallEvent::ToolCall {
            id: None,
            name: String::new(),
            arguments: String::new(),
            raw: raw.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_text_and_call_one_chunk() {
        let mut p = ToolCallParser::new();
        let evs = p.push_str("Hello <tool_call>{\"name\":\"weather\",\"arguments\":{\"loc\":\"Tokyo\"}}</tool_call> done");
        assert_eq!(evs.len(), 3);
        assert_eq!(evs[0], ToolCallEvent::Text("Hello ".to_string()));
        match &evs[1] {
            ToolCallEvent::ToolCall { name, arguments, .. } => {
                assert_eq!(name, "weather");
                assert!(arguments.contains("Tokyo"));
            }
            _ => panic!("expected ToolCall"),
        }
        assert_eq!(evs[2], ToolCallEvent::Text(" done".to_string()));
        let tail = p.finish();
        assert!(tail.is_empty());
    }

    #[test]
    fn split_across_chunks_with_partial_tag() {
        let mut p = ToolCallParser::new();
        let mut all = Vec::new();
        all.extend(p.push_str("Hello <too"));
        all.extend(p.push_str("l_call>{\"name\":\"foo\""));
        all.extend(p.push_str(",\"arguments\":{}}</tool_"));
        all.extend(p.push_str("call> bye"));
        all.extend(p.finish());
        assert!(all.iter().any(|e| matches!(e, ToolCallEvent::Text(t) if t == "Hello ")));
        assert!(all.iter().any(|e| matches!(e, ToolCallEvent::ToolCall { name, .. } if name == "foo")));
        assert!(all.iter().any(|e| matches!(e, ToolCallEvent::Text(t) if t == " bye")));
    }

    #[test]
    fn handles_utf8_boundary_split() {
        let mut p = ToolCallParser::new();
        let s = "Привет 👋 <tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>!";
        let bytes = s.as_bytes();
        let mut events = Vec::new();
        for chunk in bytes.chunks(3) {
            events.extend(p.push_bytes(chunk));
        }
        events.extend(p.finish());
        assert!(events.iter().any(|e| matches!(e, ToolCallEvent::ToolCall { name, .. } if name == "x")));
        let text: String = events
            .iter()
            .filter_map(|e| match e {
                ToolCallEvent::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert!(text.contains("Привет"));
        assert!(text.contains("👋"));
        assert!(text.ends_with('!'));
    }

    #[test]
    fn no_tool_call_returns_text_only() {
        let mut p = ToolCallParser::new();
        let evs = p.push_str("just text");
        assert_eq!(evs, vec![ToolCallEvent::Text("just text".into())]);
        let tail = p.finish();
        assert!(tail.is_empty());
    }

    #[test]
    fn open_without_close_at_finish() {
        let mut p = ToolCallParser::new();
        let head = p.push_str("a <tool_call>{name:partial");
        let tail = p.finish();
        let mut all = head;
        all.extend(tail);
        let merged: String = all
            .into_iter()
            .map(|e| match e {
                ToolCallEvent::Text(t) => t,
                ToolCallEvent::ToolCall { raw, .. } => raw,
            })
            .collect();
        assert!(merged.contains("a "));
        assert!(merged.contains("<tool_call>"));
        assert!(merged.contains("{name:partial"));
    }
}

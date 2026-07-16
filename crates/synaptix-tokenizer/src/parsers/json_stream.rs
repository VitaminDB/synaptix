use serde_json::Value as JsonValue;

use crate::error::{Result, TokenizerError};

#[derive(Debug, Clone, PartialEq)]
pub enum JsonStreamEvent {
    Value(JsonValue),
}

#[derive(Default)]
pub struct JsonStreamParser {
    buffer: String,
    pending_bytes: Vec<u8>,
}

impl JsonStreamParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Result<Vec<JsonStreamEvent>> {
        self.pending_bytes.extend_from_slice(bytes);
        let valid_up_to = match std::str::from_utf8(&self.pending_bytes) {
            Ok(s) => s.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_up_to == 0 {
            return Ok(Vec::new());
        }
        let valid = std::str::from_utf8(&self.pending_bytes[..valid_up_to])
            .expect("valid_up_to slice must be utf-8")
            .to_owned();
        self.pending_bytes.drain(..valid_up_to);
        self.push_str(&valid)
    }

    pub fn push_str(&mut self, s: &str) -> Result<Vec<JsonStreamEvent>> {
        self.buffer.push_str(s);
        self.drain_values()
    }

    fn drain_values(&mut self) -> Result<Vec<JsonStreamEvent>> {
        let trim_start = self
            .buffer
            .as_bytes()
            .iter()
            .position(|b| !b.is_ascii_whitespace())
            .unwrap_or(self.buffer.len());
        if trim_start > 0 {
            self.buffer.drain(..trim_start);
        }
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }
        let mut out = Vec::new();
        let mut last_offset = 0usize;
        let mut de = serde_json::Deserializer::from_str(&self.buffer).into_iter::<JsonValue>();
        loop {
            match de.next() {
                Some(Ok(v)) => {
                    out.push(JsonStreamEvent::Value(v));
                    last_offset = de.byte_offset();
                }
                Some(Err(e)) => {
                    if e.is_eof() {
                        break;
                    }
                    return Err(TokenizerError::Json(e));
                }
                None => break,
            }
        }
        if last_offset > 0 {
            self.buffer.drain(..last_offset);
        }
        Ok(out)
    }

    pub fn finish(mut self) -> Result<Vec<JsonStreamEvent>> {
        let mut out = self.drain_values()?;
        let leftover = self.buffer.trim();
        if !leftover.is_empty() {
            let v: JsonValue = serde_json::from_str(leftover)?;
            out.push(JsonStreamEvent::Value(v));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn stream_single_object() {
        let mut p = JsonStreamParser::new();
        let chunks = [r#"{"a":"#, r#"1,"b":2}"#];
        let mut all = Vec::new();
        for c in chunks {
            all.extend(p.push_str(c).unwrap());
        }
        all.extend(p.finish().unwrap());
        assert_eq!(all.len(), 1);
        assert_eq!(all[0], JsonStreamEvent::Value(json!({"a":1,"b":2})));
    }

    #[test]
    fn stream_multiple_values() {
        let mut p = JsonStreamParser::new();
        let s = r#"{"a":1}{"b":2}{"c":3}"#;
        let evs = p.push_str(s).unwrap();
        let tail = p.finish().unwrap();
        let total: Vec<_> = evs.into_iter().chain(tail).collect();
        assert_eq!(total.len(), 3);
        assert_eq!(total[0], JsonStreamEvent::Value(json!({"a":1})));
        assert_eq!(total[1], JsonStreamEvent::Value(json!({"b":2})));
        assert_eq!(total[2], JsonStreamEvent::Value(json!({"c":3})));
    }

    #[test]
    fn bytes_with_utf8_boundary() {
        let mut p = JsonStreamParser::new();
        let s = r#"{"name":"Привет"}{"x":1}"#;
        let bytes = s.as_bytes();
        let mut all = Vec::new();
        for chunk in bytes.chunks(3) {
            all.extend(p.push_bytes(chunk).unwrap());
        }
        all.extend(p.finish().unwrap());
        assert_eq!(all.len(), 2);
        assert_eq!(all[0], JsonStreamEvent::Value(json!({"name":"Привет"})));
    }
}

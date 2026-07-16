use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JsonState {
    ExpectStart,
    ExpectKeyStart,
    InKey,
    ExpectColon,
    ExpectValueStart,
    InString,
    InNumber,
    AfterValue,
    Done,
}

pub struct JsonSchemaConstraint {
    schema: Value,
    pub buffer: String,
    pub state: JsonState,
    pub required_keys: Vec<String>,
    pub used_keys: Vec<String>,
}

impl JsonSchemaConstraint {
    pub fn new(schema: Value) -> Self {
        let required_keys: Vec<String> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        Self {
            schema,
            buffer: String::new(),
            state: JsonState::ExpectStart,
            required_keys,
            used_keys: Vec::new(),
        }
    }

    pub fn from_str(s: &str) -> crate::error::Result<Self> {
        let v: Value = serde_json::from_str(s)
            .map_err(|e| crate::error::InferError::Other(e.to_string()))?;
        Ok(Self::new(v))
    }

    pub fn allowed_chars(&self) -> Vec<char> {
        match self.state {
            JsonState::ExpectStart => vec!['{'],
            JsonState::ExpectKeyStart => vec!['"', '}'],
            JsonState::InKey => {
                let mut c: Vec<char> = ('a'..='z').collect();
                c.extend('A'..='Z');
                c.extend('0'..='9');
                c.push('_');
                c.push('"');
                c
            }
            JsonState::ExpectColon => vec![':', ' '],
            JsonState::ExpectValueStart => vec!['"', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 't', 'f', 'n'],
            JsonState::InString => {
                let mut c: Vec<char> = (' '..='~').filter(|&x| x != '"' && x != '\\').collect();
                c.push('"');
                c.push('\\');
                c
            }
            JsonState::InNumber => {
                let mut c: Vec<char> = ('0'..='9').collect();
                c.push('.');
                c.push(',');
                c.push('}');
                c
            }
            JsonState::AfterValue => vec![',', '}'],
            JsonState::Done => Vec::new(),
        }
    }

    pub fn advance(&mut self, ch: char) -> bool {
        match self.state {
            JsonState::ExpectStart => {
                if ch == '{' { self.state = JsonState::ExpectKeyStart; true } else { false }
            }
            JsonState::ExpectKeyStart => {
                if ch == '"' { self.buffer.clear(); self.state = JsonState::InKey; true }
                else if ch == '}' { self.state = JsonState::Done; true }
                else { false }
            }
            JsonState::InKey => {
                if ch == '"' {
                    self.used_keys.push(self.buffer.clone());
                    self.state = JsonState::ExpectColon;
                    true
                } else {
                    self.buffer.push(ch);
                    true
                }
            }
            JsonState::ExpectColon => {
                if ch == ':' { self.state = JsonState::ExpectValueStart; true }
                else if ch == ' ' { true }
                else { false }
            }
            JsonState::ExpectValueStart => {
                if ch == '"' { self.state = JsonState::InString; true }
                else if ch.is_ascii_digit() || ch == '-' { self.state = JsonState::InNumber; true }
                else { false }
            }
            JsonState::InString => {
                if ch == '"' { self.state = JsonState::AfterValue; true }
                else { true }
            }
            JsonState::InNumber => {
                if ch.is_ascii_digit() || ch == '.' { true }
                else if ch == ',' { self.state = JsonState::ExpectKeyStart; true }
                else if ch == '}' {
                    if self.all_required_used() { self.state = JsonState::Done; true } else { false }
                }
                else { false }
            }
            JsonState::AfterValue => {
                if ch == ',' { self.state = JsonState::ExpectKeyStart; true }
                else if ch == '}' {
                    if self.all_required_used() { self.state = JsonState::Done; true } else { false }
                }
                else { false }
            }
            JsonState::Done => false,
        }
    }

    fn all_required_used(&self) -> bool {
        self.required_keys.iter().all(|k| self.used_keys.contains(k))
    }

    pub fn is_done(&self) -> bool {
        self.state == JsonState::Done
    }

    pub fn schema(&self) -> &Value {
        &self.schema
    }

    pub fn allowed_tokens(&self, _vocab_size: usize) -> Vec<u32> {
        Vec::new()
    }
}

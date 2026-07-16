use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningEvent {
    Visible(String),
    Thinking(String),
}

pub struct ReasoningStreamParser {
    start_tag: String,
    end_tag: String,
    state: State,
    buffer: String,
    pending_bytes: Vec<u8>,
}

#[derive(Debug)]
enum State {
    Visible,
    Thinking,
}

impl Default for ReasoningStreamParser {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningStreamParser {
    pub fn new() -> Self {
        Self::with_tags("<think>", "</think>")
    }

    pub fn with_tags(start: impl Into<String>, end: impl Into<String>) -> Self {
        Self {
            start_tag: start.into(),
            end_tag: end.into(),
            state: State::Visible,
            buffer: String::new(),
            pending_bytes: Vec::new(),
        }
    }

    pub fn push_bytes(&mut self, bytes: &[u8]) -> Vec<ReasoningEvent> {
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

    pub fn push_str(&mut self, s: &str) -> Vec<ReasoningEvent> {
        self.buffer.push_str(s);
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Visible => {
                    if let Some(pos) = self.buffer.find(&self.start_tag) {
                        if pos > 0 {
                            out.push(ReasoningEvent::Visible(self.buffer[..pos].to_string()));
                        }
                        let after = pos + self.start_tag.len();
                        self.buffer.drain(..after);
                        self.state = State::Thinking;
                        continue;
                    }
                    let safe = safe_emit_len(&self.buffer, &self.start_tag);
                    if safe > 0 {
                        out.push(ReasoningEvent::Visible(self.buffer[..safe].to_string()));
                        self.buffer.drain(..safe);
                    }
                    break;
                }
                State::Thinking => {
                    if let Some(pos) = self.buffer.find(&self.end_tag) {
                        if pos > 0 {
                            out.push(ReasoningEvent::Thinking(self.buffer[..pos].to_string()));
                        }
                        let after = pos + self.end_tag.len();
                        self.buffer.drain(..after);
                        self.state = State::Visible;
                        continue;
                    }
                    let safe = safe_emit_len(&self.buffer, &self.end_tag);
                    if safe > 0 {
                        out.push(ReasoningEvent::Thinking(self.buffer[..safe].to_string()));
                        self.buffer.drain(..safe);
                    }
                    break;
                }
            }
        }
        out
    }

    pub fn finish(mut self) -> Vec<ReasoningEvent> {
        let mut out = Vec::new();
        if !self.buffer.is_empty() {
            let leftover = std::mem::take(&mut self.buffer);
            match self.state {
                State::Visible => out.push(ReasoningEvent::Visible(leftover)),
                State::Thinking => out.push(ReasoningEvent::Thinking(leftover)),
            }
        }
        if !self.pending_bytes.is_empty() {
            out.push(ReasoningEvent::Visible(
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
    let mut cut = buf.len().saturating_sub(max_k);
    while cut > 0 && !buf.is_char_boundary(cut) {
        cut -= 1;
    }
    cut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_blocks() {
        let mut p = ReasoningStreamParser::new();
        let evs = p.push_str("ok<think>secret reasoning</think>final answer");
        let tail = p.finish();
        let all: Vec<_> = evs.into_iter().chain(tail).collect();
        assert_eq!(all.len(), 3);
        assert_eq!(all[0], ReasoningEvent::Visible("ok".into()));
        assert_eq!(all[1], ReasoningEvent::Thinking("secret reasoning".into()));
        assert_eq!(all[2], ReasoningEvent::Visible("final answer".into()));
    }

    #[test]
    fn chunked_across_tags() {
        let mut p = ReasoningStreamParser::new();
        let mut all = Vec::new();
        for c in ["a<thi", "nk>b</thi", "nk>c"] {
            all.extend(p.push_str(c));
        }
        all.extend(p.finish());
        let visible: Vec<_> = all
            .iter()
            .filter_map(|e| match e {
                ReasoningEvent::Visible(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        let thinking: Vec<_> = all
            .iter()
            .filter_map(|e| match e {
                ReasoningEvent::Thinking(s) => Some(s.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(visible.concat(), "ac");
        assert_eq!(thinking.concat(), "b");
    }
}

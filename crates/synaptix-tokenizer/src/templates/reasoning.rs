use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReasoningConfig {
    pub think_start: String,
    pub think_end: String,
    pub strip_reasoning_in_history: bool,
}

impl ReasoningConfig {
    pub fn qwen3() -> Self {
        Self {
            think_start: "<think>".into(),
            think_end: "</think>".into(),
            strip_reasoning_in_history: true,
        }
    }

    pub fn deepseek_r1() -> Self {
        Self {
            think_start: "<think>".into(),
            think_end: "</think>".into(),
            strip_reasoning_in_history: true,
        }
    }

    pub fn strip(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        while let Some(start) = rest.find(self.think_start.as_str()) {
            out.push_str(&rest[..start]);
            let after_open = &rest[start + self.think_start.len()..];
            if let Some(end) = after_open.find(self.think_end.as_str()) {
                rest = &after_open[end + self.think_end.len()..];
            } else {
                rest = "";
                break;
            }
        }
        out.push_str(rest);
        out
    }
}

impl Default for ReasoningConfig {
    fn default() -> Self {
        Self::qwen3()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_block() {
        let r = ReasoningConfig::qwen3();
        let s = r.strip("hello <think>x y z</think>world");
        assert_eq!(s, "hello world");
    }

    #[test]
    fn strip_multiple_blocks() {
        let r = ReasoningConfig::qwen3();
        let s = r.strip("a<think>b</think>c<think>d</think>e");
        assert_eq!(s, "ace");
    }

    #[test]
    fn strip_unclosed_drops_tail() {
        let r = ReasoningConfig::qwen3();
        let s = r.strip("a<think>b");
        assert_eq!(s, "a");
    }
}

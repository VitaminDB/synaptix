pub struct LcpDelta {
    previous: String,
}

impl Default for LcpDelta {
    fn default() -> Self {
        Self::new()
    }
}

impl LcpDelta {
    pub fn new() -> Self {
        Self { previous: String::new() }
    }

    pub fn ingest(&mut self, current: &str) -> (String, String) {
        let common = lcp_chars(&self.previous, current);
        let prev_to_delete = self.previous[common..].to_string();
        let to_append = current[common..].to_string();
        self.previous = current.to_string();
        (prev_to_delete, to_append)
    }

    pub fn reset(&mut self) {
        self.previous.clear();
    }
}

fn lcp_chars(a: &str, b: &str) -> usize {
    let mut idx = 0;
    let mut ai = a.char_indices();
    let mut bi = b.char_indices();
    loop {
        match (ai.next(), bi.next()) {
            (Some((ai_pos, ac)), Some((_bi_pos, bc))) if ac == bc => {
                idx = ai_pos + ac.len_utf8();
            }
            _ => break,
        }
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lcp_delta_appends_when_prefix_matches() {
        let mut d = LcpDelta::new();
        let (del, add) = d.ingest("hello");
        assert_eq!(del, "");
        assert_eq!(add, "hello");
        let (del, add) = d.ingest("hello world");
        assert_eq!(del, "");
        assert_eq!(add, " world");
    }

    #[test]
    fn lcp_delta_replaces_suffix_on_revision() {
        let mut d = LcpDelta::new();
        let _ = d.ingest("hello there");
        let (del, add) = d.ingest("hello world");
        assert_eq!(del, "there");
        assert_eq!(add, "world");
    }

    #[test]
    fn lcp_delta_handles_utf8() {
        let mut d = LcpDelta::new();
        let _ = d.ingest("привет");
        let (del, add) = d.ingest("привет мир");
        assert_eq!(del, "");
        assert_eq!(add, " мир");
    }
}

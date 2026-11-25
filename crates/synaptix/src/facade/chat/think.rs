const OPEN: &str = "<think>";
const CLOSE: &str = "</think>";

/// Теги блока размышлений модели. Обычно `<think>`/`</think>`; у Hy4-flash —
/// `<think:opensource>`/`</think:opensource>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThinkTags {
    pub open: &'static str,
    pub close: &'static str,
    /// Что дописать, закрывая размышления принудительно (бюджет).
    pub forced: &'static str,
}

pub const DEFAULT_TAGS: ThinkTags = ThinkTags { open: OPEN, close: CLOSE, forced: "\n</think>\n\n" };
pub const HY4_TAGS: ThinkTags = ThinkTags { open: "<think:opensource>", close: "</think:opensource>", forced: "\n</think:opensource>" };

impl ThinkTags {
    /// Теги по словарю: чей закрывающий тег — один токен (`single`).
    pub fn detect(single: impl Fn(&str) -> bool) -> Self {
        if single(HY4_TAGS.close) { HY4_TAGS } else { DEFAULT_TAGS }
    }
}

impl Default for ThinkTags {
    fn default() -> Self {
        DEFAULT_TAGS
    }
}

#[derive(Default, Debug)]
pub struct ThinkParser {
    tags: ThinkTags,
    inside: bool,
    pending: String,
    eat_extraneous_close: bool,
    /// Блок открыт шаблоном, но модель (Qwen3) всё равно печатает `<think>`:
    /// ведущий тег — не текст размышлений.
    skip_open: bool,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ThinkSplit {
    pub body: String,
    pub thinking: String,
}

impl ThinkParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_implicit_open() -> Self {
        Self {
            tags: DEFAULT_TAGS,
            inside: true,
            pending: String::new(),
            eat_extraneous_close: true,
            skip_open: true,
        }
    }

    /// Другие теги блока (см. [`ThinkTags::detect`]).
    pub fn with_tags(mut self, tags: ThinkTags) -> Self {
        self.tags = tags;
        self
    }

    pub fn tags(&self) -> ThinkTags {
        self.tags
    }

    pub fn feed(&mut self, chunk: &str) -> ThinkSplit {
        let (open, close) = (self.tags.open, self.tags.close);
        let mut buf = std::mem::take(&mut self.pending);
        buf.push_str(chunk);
        if self.skip_open {
            let t = buf.trim_start();
            let lower = t.to_ascii_lowercase();
            if open.starts_with(lower.as_str()) && lower.len() < open.len() {
                // Пусто или начало тега — ждём следующий кусок.
                self.pending = buf;
                return ThinkSplit::default();
            }
            if lower.starts_with(open) {
                buf = t[open.len()..].to_string();
            }
            self.skip_open = false;
        }

        let mut out = ThinkSplit::default();
        let mut cursor = 0usize;

        loop {
            if self.inside {
                match find_ci(&buf, close, cursor) {
                    Some(pos) => {
                        push_segment(&mut out, true, &buf[cursor..pos]);
                        cursor = pos + close.len();
                        self.inside = false;
                    }
                    None => {
                        let split = trailing_tag_prefix(&buf[cursor..], close);
                        let safe_end = cursor + split;
                        push_segment(&mut out, true, &buf[cursor..safe_end]);
                        self.pending = buf[safe_end..].to_string();
                        return out;
                    }
                }
            } else {
                let open_pos = find_ci(&buf, open, cursor);
                let close_pos = if self.eat_extraneous_close {
                    find_ci(&buf, close, cursor)
                } else {
                    None
                };

                let (next_pos, tag_len, opens) = match (open_pos, close_pos) {
                    (None, None) => {
                        let split_open = trailing_tag_prefix(&buf[cursor..], open);
                        let split = if self.eat_extraneous_close {
                            let split_close = trailing_tag_prefix(&buf[cursor..], close);
                            split_open.min(split_close)
                        } else {
                            split_open
                        };
                        let safe_end = cursor + split;
                        push_segment(&mut out, false, &buf[cursor..safe_end]);
                        self.pending = buf[safe_end..].to_string();
                        return out;
                    }
                    (Some(o), None) => (o, open.len(), true),
                    (None, Some(c)) => (c, close.len(), false),
                    (Some(o), Some(c)) => {
                        if o <= c {
                            (o, open.len(), true)
                        } else {
                            (c, close.len(), false)
                        }
                    }
                };

                push_segment(&mut out, false, &buf[cursor..next_pos]);
                cursor = next_pos + tag_len;
                if opens {
                    self.inside = true;
                }
            }
        }
    }
}

fn push_segment(out: &mut ThinkSplit, inside: bool, segment: &str) {
    if segment.is_empty() {
        return;
    }
    if inside {
        out.thinking.push_str(segment);
    } else {
        out.body.push_str(segment);
    }
}

fn find_ci(haystack: &str, needle: &str, from: usize) -> Option<usize> {
    if needle.is_empty() {
        return Some(from);
    }
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    if from + n.len() > h.len() {
        return None;
    }
    let mut i = from;
    while i + n.len() <= h.len() {
        if h[i..i + n.len()]
            .iter()
            .zip(n.iter())
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

fn trailing_tag_prefix(s: &str, tag: &str) -> usize {
    let tag_bytes = tag.as_bytes();
    let s_bytes = s.as_bytes();
    let max_check = tag_bytes.len().saturating_sub(1).min(s_bytes.len());

    for i in (1..=max_check).rev() {
        let tail = &s_bytes[s_bytes.len() - i..];
        let cut = s_bytes.len() - i;
        if !s.is_char_boundary(cut) {
            continue;
        }
        if tail
            .iter()
            .zip(tag_bytes.iter())
            .all(|(a, b)| a.eq_ignore_ascii_case(b))
        {
            return cut;
        }
    }
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn split(parser: &mut ThinkParser, chunks: &[&str]) -> ThinkSplit {
        let mut acc = ThinkSplit::default();
        for c in chunks {
            let part = parser.feed(c);
            acc.body.push_str(&part.body);
            acc.thinking.push_str(&part.thinking);
        }
        acc
    }

    /// Шаблон открыл блок, а модель всё равно печатает `<think>` (Qwen3):
    /// тег не попадает в размышления, и по кускам тоже.
    #[test]
    fn implicit_open_drops_echoed_open_tag() {
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["<thi", "nk>\nдумаю", "</think>ответ"]);
        assert_eq!(r.thinking.trim(), "думаю");
        assert_eq!(r.body, "ответ");
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["думаю</think>ответ"]);
        assert_eq!(r.thinking, "думаю");
    }

    #[test]
    fn hy4_tags() {
        let mut p = ThinkParser::new_implicit_open().with_tags(HY4_TAGS);
        let r = split(&mut p, &["думаю</think:open", "source>ответ"]);
        assert_eq!(r.thinking, "думаю");
        assert_eq!(r.body, "ответ");
        assert_eq!(ThinkTags::detect(|s| s == "</think:opensource>"), HY4_TAGS);
        assert_eq!(ThinkTags::detect(|_| false), DEFAULT_TAGS);
    }

    #[test]
    fn whole_block_in_one_chunk() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["<think>a</think>b"]);
        assert_eq!(r.body, "b");
        assert_eq!(r.thinking, "a");
    }

    #[test]
    fn body_only_no_tags() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["hello world"]);
        assert_eq!(r.body, "hello world");
        assert_eq!(r.thinking, "");
    }

    #[test]
    fn thinking_only_complete() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["<think>x</think>"]);
        assert_eq!(r.body, "");
        assert_eq!(r.thinking, "x");
    }

    #[test]
    fn open_tag_split_across_chunks() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["<thi", "nk>x</thi", "nk>y"]);
        assert_eq!(r.body, "y");
        assert_eq!(r.thinking, "x");
    }

    #[test]
    fn close_tag_split_at_last_byte() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["<think>abc</think", ">tail"]);
        assert_eq!(r.body, "tail");
        assert_eq!(r.thinking, "abc");
    }

    #[test]
    fn open_without_close_streams_thinking_so_far() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["<think>abc"]);
        assert_eq!(r.body, "");
        assert_eq!(r.thinking, "abc");
    }

    #[test]
    fn case_insensitive_tags() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["<THINK>X</Think>Y"]);
        assert_eq!(r.body, "Y");
        assert_eq!(r.thinking, "X");
    }

    #[test]
    fn body_then_thinking_then_body() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["pre<think>mid</think>post"]);
        assert_eq!(r.body, "prepost");
        assert_eq!(r.thinking, "mid");
    }

    #[test]
    fn pending_does_not_eat_short_lookalike() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["abc<", "def"]);
        assert_eq!(r.body, "abc<def");
        assert_eq!(r.thinking, "");
    }

    #[test]
    fn utf8_boundary_safe() {
        let mut p = ThinkParser::new();
        let r = split(&mut p, &["приве", "т<think>да</think>!"]);
        assert_eq!(r.body, "привет!");
        assert_eq!(r.thinking, "да");
    }

    #[test]
    fn implicit_open_thinking_then_close_then_body() {
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["рассужден", "ия про X</think>финальный ответ"]);
        assert_eq!(r.thinking, "рассуждения про X");
        assert_eq!(r.body, "финальный ответ");
    }

    #[test]
    fn implicit_open_double_close_eaten() {
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["abc</think>\n</think>tail"]);
        assert_eq!(r.thinking, "abc");
        assert_eq!(r.body, "\ntail");
    }

    #[test]
    fn implicit_open_no_close_streams_thinking_only() {
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["размышляю про", " что-то"]);
        assert_eq!(r.thinking, "размышляю про что-то");
        assert_eq!(r.body, "");
    }

    #[test]
    fn implicit_open_explicit_reopen_after_close() {
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["mid</think>body<think>more</think>tail"]);
        assert_eq!(r.thinking, "midmore");
        assert_eq!(r.body, "bodytail");
    }

    #[test]
    fn implicit_open_split_close_across_chunks() {
        let mut p = ThinkParser::new_implicit_open();
        let r = split(&mut p, &["abc</thi", "nk>tail"]);
        assert_eq!(r.thinking, "abc");
        assert_eq!(r.body, "tail");
    }

    #[test]
    fn implicit_open_extraneous_close_in_separate_feed() {
        let mut p = ThinkParser::new_implicit_open();
        let r1 = p.feed("abc</think>");
        assert_eq!(r1.thinking, "abc");
        assert_eq!(r1.body, "");
        let r2 = p.feed("</think>tail");
        assert_eq!(r2.thinking, "");
        assert_eq!(r2.body, "tail");
    }
}

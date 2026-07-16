use super::chunking::{chunk_by_chars, TextChunk};

pub fn html_to_text(html: &str) -> String {
    htmd::convert(html).unwrap_or_else(|_| strip_tags(html))
}

pub fn html_to_chunks(html: &str, max_chars: usize, overlap: usize) -> Vec<TextChunk> {
    let text = html_to_text(html);
    chunk_by_chars(&text, max_chars, overlap)
}

fn strip_tags(html: &str) -> String {
    let mut out = String::with_capacity(html.len());
    let mut in_tag = false;
    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ => {
                if !in_tag {
                    out.push(ch);
                }
            }
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

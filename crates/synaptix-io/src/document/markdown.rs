use pulldown_cmark::{Event, Parser, Tag, TagEnd};

use super::chunking::{chunk_by_chars, TextChunk};

pub fn markdown_to_text(md: &str) -> String {
    let parser = Parser::new(md);
    let mut out = String::new();
    let mut in_code_block = false;

    for event in parser {
        match event {
            Event::Text(t) => {
                if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') {
                    out.push(' ');
                }
                out.push_str(&t);
            }
            Event::Code(t) => {
                if !out.is_empty() && !out.ends_with(' ') {
                    out.push(' ');
                }
                out.push_str(&t);
            }
            Event::Start(Tag::CodeBlock(_)) => {
                in_code_block = true;
            }
            Event::End(TagEnd::CodeBlock) => {
                in_code_block = false;
            }
            Event::SoftBreak | Event::HardBreak => {
                out.push(' ');
            }
            _ => {}
        }
    }
    let _ = in_code_block;
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

pub fn markdown_to_chunks(md: &str, max_chars: usize, overlap: usize) -> Vec<TextChunk> {
    let text = markdown_to_text(md);
    chunk_by_chars(&text, max_chars, overlap)
}

pub fn extract_headings(md: &str) -> Vec<(u32, String)> {
    let parser = Parser::new(md);
    let mut headings = Vec::new();
    let mut current_level: Option<u32> = None;
    let mut current_text = String::new();

    for event in parser {
        match event {
            Event::Start(Tag::Heading { level, .. }) => {
                current_level = Some(level as u32);
                current_text.clear();
            }
            Event::Text(t) => {
                if current_level.is_some() {
                    current_text.push_str(&t);
                }
            }
            Event::End(TagEnd::Heading(_)) => {
                if let Some(level) = current_level.take() {
                    headings.push((level, current_text.trim().to_string()));
                    current_text.clear();
                }
            }
            _ => {}
        }
    }

    headings
}

//! Markdown-aware сплиттер: режем документ на блоки по структуре (заголовки
//! `#`, огороженные блоки кода ```), затем каждую секцию, превышающую
//! chunk_size, добиваем рекурсивным сплиттером.

use crate::chunking::recursive::recursive_chunk;

pub fn markdown_aware_chunk(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return Vec::new();
    }
    let sections = split_markdown_sections(text);
    let mut out = Vec::new();
    for sec in sections {
        if sec.trim().is_empty() {
            continue;
        }
        if sec.chars().count() <= chunk_size {
            out.push(sec);
        } else {
            out.extend(recursive_chunk(&sec, chunk_size, overlap));
        }
    }
    out
}

/// Разбить на секции: каждый заголовок (`#`..`######`) начинает новую секцию;
/// огороженные блоки кода (```) не разрезаются.
fn split_markdown_sections(text: &str) -> Vec<String> {
    let mut sections: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_code = false;

    for line in text.lines() {
        let trimmed = line.trim_start();
        let is_fence = trimmed.starts_with("```");
        let is_heading = !in_code && is_markdown_heading(trimmed);

        if is_heading && !cur.is_empty() {
            sections.push(std::mem::take(&mut cur));
        }

        cur.push_str(line);
        cur.push('\n');

        if is_fence {
            in_code = !in_code;
        }
    }
    if !cur.is_empty() {
        sections.push(cur);
    }
    sections
}

fn is_markdown_heading(trimmed: &str) -> bool {
    let hashes = trimmed.chars().take_while(|&c| c == '#').count();
    (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ')
}

//! Рекурсивный сплиттер по убывающим разделителям (стиль LangChain
//! RecursiveCharacterTextSplitter): режем по `\n\n`, затем `\n`, `. `, ` `, и в
//! крайнем случае по символам, после чего жадно склеиваем мелкие куски в чанки
//! ≤ chunk_size с перекрытием.

const SEPARATORS: &[&str] = &["\n\n", "\n", ". ", " ", ""];

pub fn recursive_chunk(text: &str, chunk_size: usize, overlap: usize) -> Vec<String> {
    if text.is_empty() || chunk_size == 0 {
        return Vec::new();
    }
    let pieces = split_recursive(text, chunk_size, 0);
    merge_with_overlap(pieces, chunk_size, overlap)
}

/// Разбить на атомарные куски ≤ chunk_size (по возможности), спускаясь по
/// списку разделителей.
fn split_recursive(text: &str, chunk_size: usize, sep_idx: usize) -> Vec<String> {
    if text.chars().count() <= chunk_size {
        return if text.is_empty() { Vec::new() } else { vec![text.to_string()] };
    }
    if sep_idx >= SEPARATORS.len() || SEPARATORS[sep_idx].is_empty() {
        return hard_split(text, chunk_size);
    }
    let sep = SEPARATORS[sep_idx];
    let mut out = Vec::new();
    for part in split_keep_sep(text, sep) {
        if part.chars().count() <= chunk_size {
            if !part.is_empty() {
                out.push(part);
            }
        } else {
            out.extend(split_recursive(&part, chunk_size, sep_idx + 1));
        }
    }
    out
}

/// Разбить по разделителю, оставляя его в конце предыдущего куска (склейка
/// восстанавливает исходную пунктуацию/переводы строк).
fn split_keep_sep(text: &str, sep: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text;
    while let Some(pos) = rest.find(sep) {
        let end = pos + sep.len();
        out.push(rest[..end].to_string());
        rest = &rest[end..];
    }
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

fn hard_split(text: &str, chunk_size: usize) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let end = (i + chunk_size).min(chars.len());
        out.push(chars[i..end].iter().collect());
        i = end;
    }
    out
}

/// Жадно склеить куски (каждый ≤ chunk_size) в чанки ≤ chunk_size. Перед новым
/// чанком подмешиваем хвост предыдущего длиной `overlap` — но только если он
/// помещается вместе с первым куском, чтобы строго не превысить chunk_size.
fn merge_with_overlap(pieces: Vec<String>, chunk_size: usize, overlap: usize) -> Vec<String> {
    let mut chunks: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_len = 0usize;
    let mut pending_overlap = String::new();

    for p in pieces {
        let plen = p.chars().count();
        // Закрыть текущий чанк, если кусок не влезает.
        if cur_len + plen > chunk_size && cur_len > 0 {
            chunks.push(cur.clone());
            pending_overlap = if overlap > 0 {
                cur.chars().skip(cur_len.saturating_sub(overlap)).collect()
            } else {
                String::new()
            };
            cur.clear();
            cur_len = 0;
        }
        // Засеять новый чанк перекрытием, если оно помещается вместе с куском.
        if cur_len == 0 && !pending_overlap.is_empty() {
            let olen = pending_overlap.chars().count();
            if olen + plen <= chunk_size {
                cur.push_str(&pending_overlap);
                cur_len += olen;
            }
            pending_overlap.clear();
        }
        cur.push_str(&p);
        cur_len += plen;
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    chunks
}

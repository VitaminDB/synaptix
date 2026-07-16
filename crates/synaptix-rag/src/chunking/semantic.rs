//! Семантический сплиттер: бьём текст на предложения, эмбеддим их и начинаем
//! новый чанк там, где косинусная близость соседних предложений падает ниже
//! порога (подход Greg Kamradt semantic chunking).

use synaptix_core::dtype::DType;
use crate::embedder::Embedder;
use crate::error::{RagError, Result};
use crate::metric::cosine;

/// Разбить `text` на предложения и сгруппировать по семантической близости.
/// `threshold` — минимальная косинусная близость соседних предложений, при
/// которой они остаются в одном чанке.
pub fn semantic_chunk(text: &str, embedder: &dyn Embedder, threshold: f32) -> Result<Vec<String>> {
    let sentences = split_sentences(text);
    if sentences.len() <= 1 {
        return Ok(sentences);
    }
    let emb = embedder.embed(&sentences)?;
    let rows: Vec<Vec<f32>> = emb.to_dtype(DType::F32).and_then(|t| t.to_vec2::<f32>()).map_err(RagError::Core)?;
    if rows.len() != sentences.len() {
        return Err(RagError::Other(format!(
            "embedder returned {} rows for {} sentences",
            rows.len(),
            sentences.len()
        )));
    }

    let mut chunks = Vec::new();
    let mut cur = sentences[0].clone();
    for i in 1..sentences.len() {
        let sim = cosine(&rows[i - 1], &rows[i]);
        if sim < threshold {
            chunks.push(std::mem::take(&mut cur));
            cur = sentences[i].clone();
        } else {
            cur.push(' ');
            cur.push_str(&sentences[i]);
        }
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    Ok(chunks)
}

/// Простое деление на предложения по терминаторам `.`, `!`, `?` и переводам строк.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?' | '\n') {
            let s = cur.trim();
            if !s.is_empty() {
                out.push(s.to_string());
            }
            cur.clear();
        }
    }
    let s = cur.trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
    out
}

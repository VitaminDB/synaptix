use std::path::Path;

use crate::error::{IoError, Result};
use super::chunking::{chunk_by_chars, TextChunk};

pub fn pdf_to_text(path: impl AsRef<Path>) -> Result<String> {
    pdf_extract::extract_text(path.as_ref())
        .map_err(|e| IoError::Document(format!("pdf_extract: {e}")))
}

pub fn pdf_to_chunks(path: impl AsRef<Path>, max_chars: usize, overlap: usize) -> Result<Vec<TextChunk>> {
    let text = pdf_to_text(path)?;
    Ok(chunk_by_chars(&text, max_chars, overlap))
}

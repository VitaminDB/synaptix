//! Минимальный SentencePiece-декодер: парсит `tokenizer.model` (protobuf
//! `ModelProto`) — извлекает только список piece-строк — и реализует CTC-decode
//! `token_ids -> text` (конкатенация pieces, `▁` U+2581 → пробел, обрезка ведущего
//! пробела). Полный SP-кодировщик не нужен (нам требуется только `decode`).
//!
//! Формат protobuf `ModelProto`:
//!   field 1 (repeated, wire-type 2): `SentencePiece { string piece = 1; ... }`.
//! Парсим верхнеуровневое поле 1, внутри — вложенное поле 1 (piece-строка).

use crate::GigaAmError;

const SPACE_MARK: char = '\u{2581}'; // ▁

pub struct SpmDecoder {
    pieces: Vec<String>,
}

impl SpmDecoder {
    pub fn from_model_bytes(bytes: &[u8]) -> Result<Self, GigaAmError> {
        let mut pieces = Vec::new();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let (field, wire, p) = read_key(bytes, pos)
                .ok_or_else(|| GigaAmError::Tokenizer("spm: truncated key".into()))?;
            pos = p;
            match (field, wire) {
                // pieces (repeated message).
                (1, 2) => {
                    let (msg, p) = read_len_delimited(bytes, pos)
                        .ok_or_else(|| GigaAmError::Tokenizer("spm: truncated piece msg".into()))?;
                    pos = p;
                    let piece = parse_piece(msg)?;
                    pieces.push(piece);
                }
                // прочие поля верхнего уровня — пропускаем по wire-type.
                _ => {
                    pos = skip_field(bytes, pos, wire)
                        .ok_or_else(|| GigaAmError::Tokenizer("spm: skip field".into()))?;
                }
            }
        }
        if pieces.is_empty() {
            return Err(GigaAmError::Tokenizer("spm: no pieces parsed".into()));
        }
        Ok(Self { pieces })
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn id_to_piece(&self, id: usize) -> Option<&str> {
        self.pieces.get(id).map(|s| s.as_str())
    }

    /// Конкатенация pieces по id, `▁` → пробел, обрезка ведущего пробела.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(p) = self.pieces.get(id as usize) {
                for ch in p.chars() {
                    if ch == SPACE_MARK {
                        out.push(' ');
                    } else {
                        out.push(ch);
                    }
                }
            }
        }
        out.trim_start().to_string()
    }
}

/// Внутри `SentencePiece`-сообщения берём field 1 (piece-строку).
fn parse_piece(msg: &[u8]) -> Result<String, GigaAmError> {
    let mut pos = 0usize;
    while pos < msg.len() {
        let (field, wire, p) = read_key(msg, pos)
            .ok_or_else(|| GigaAmError::Tokenizer("spm: truncated piece key".into()))?;
        pos = p;
        if field == 1 && wire == 2 {
            let (s, _) = read_len_delimited(msg, pos)
                .ok_or_else(|| GigaAmError::Tokenizer("spm: truncated piece str".into()))?;
            return String::from_utf8(s.to_vec())
                .map_err(|e| GigaAmError::Tokenizer(format!("spm: piece utf8: {e}")));
        }
        pos = skip_field(msg, pos, wire)
            .ok_or_else(|| GigaAmError::Tokenizer("spm: skip piece field".into()))?;
    }
    Err(GigaAmError::Tokenizer("spm: piece missing field 1".into()))
}

/// (field_number, wire_type, new_pos).
fn read_key(buf: &[u8], pos: usize) -> Option<(u64, u8, usize)> {
    let (key, p) = read_varint(buf, pos)?;
    Some((key >> 3, (key & 0x7) as u8, p))
}

fn read_varint(buf: &[u8], mut pos: usize) -> Option<(u64, usize)> {
    let mut result = 0u64;
    let mut shift = 0u32;
    loop {
        let byte = *buf.get(pos)?;
        pos += 1;
        result |= ((byte & 0x7f) as u64) << shift;
        if byte & 0x80 == 0 {
            return Some((result, pos));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

fn read_len_delimited(buf: &[u8], pos: usize) -> Option<(&[u8], usize)> {
    let (len, p) = read_varint(buf, pos)?;
    let len = len as usize;
    let end = p.checked_add(len)?;
    if end > buf.len() {
        return None;
    }
    Some((&buf[p..end], end))
}

fn skip_field(buf: &[u8], pos: usize, wire: u8) -> Option<usize> {
    match wire {
        0 => read_varint(buf, pos).map(|(_, p)| p),
        1 => pos.checked_add(8).filter(|&e| e <= buf.len()),
        2 => read_len_delimited(buf, pos).map(|(_, p)| p),
        5 => pos.checked_add(4).filter(|&e| e <= buf.len()),
        _ => None,
    }
}

use once_cell::sync::Lazy;
use std::collections::HashMap;

use crate::error::{Result, TokenizerError};

pub static BYTE_TO_CHAR: Lazy<[char; 256]> = Lazy::new(|| {
    let mut byte_to_unicode = [0u32; 256];
    let mut visible: Vec<u8> = Vec::with_capacity(256);
    for b in b'!'..=b'~' {
        visible.push(b);
    }
    for b in 0xA1u8..=0xACu8 {
        visible.push(b);
    }
    for b in 0xAEu8..=0xFFu8 {
        visible.push(b);
    }
    for &b in &visible {
        byte_to_unicode[b as usize] = b as u32;
    }
    let mut n: u32 = 0;
    for b in 0u16..=255u16 {
        if !visible.contains(&(b as u8)) {
            byte_to_unicode[b as usize] = 256 + n;
            n += 1;
        }
    }
    let mut result = ['\0'; 256];
    for (i, &cp) in byte_to_unicode.iter().enumerate() {
        result[i] = char::from_u32(cp).expect("byte_to_unicode mapping must yield valid char");
    }
    result
});

pub static CHAR_TO_BYTE: Lazy<HashMap<char, u8>> = Lazy::new(|| {
    let mut m = HashMap::with_capacity(256);
    for (b, c) in BYTE_TO_CHAR.iter().enumerate() {
        m.insert(*c, b as u8);
    }
    m
});

pub fn bytes_to_string(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len());
    for &b in bytes {
        s.push(BYTE_TO_CHAR[b as usize]);
    }
    s
}

pub fn string_to_bytes(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for c in s.chars() {
        let b = CHAR_TO_BYTE.get(&c).ok_or_else(|| {
            TokenizerError::InvalidArgument(format!(
                "char `{}` (U+{:04X}) is not in the GPT-2 byte-level table",
                c, c as u32
            ))
        })?;
        out.push(*b);
    }
    Ok(out)
}

pub fn space_marker() -> char {
    BYTE_TO_CHAR[b' ' as usize]
}

pub fn add_prefix_space(text: &str) -> String {
    if text.starts_with(' ') {
        text.to_string()
    } else {
        let mut out = String::with_capacity(text.len() + 1);
        out.push(' ');
        out.push_str(text);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn space_maps_to_g_tilde() {
        assert_eq!(space_marker(), 'Ġ');
    }

    fn roundtrip_text(text: &str) {
        let bytes = text.as_bytes();
        let encoded = bytes_to_string(bytes);
        let decoded = string_to_bytes(&encoded).unwrap();
        assert_eq!(decoded.as_slice(), bytes);
    }

    #[test]
    fn roundtrip_ascii() {
        roundtrip_text("Hello, world!");
    }

    #[test]
    fn roundtrip_cyrillic() {
        roundtrip_text("Привет, мир!");
    }

    #[test]
    fn roundtrip_emoji() {
        roundtrip_text("hello 👋🌍 multi-byte");
    }

    #[test]
    fn all_bytes_unique() {
        let mut seen = std::collections::HashSet::new();
        for c in BYTE_TO_CHAR.iter() {
            assert!(seen.insert(*c), "duplicate char in mapping");
        }
        assert_eq!(seen.len(), 256);
    }
}

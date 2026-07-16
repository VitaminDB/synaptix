use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::Tokenizer as HfInner;

use crate::bpe::serialize_tokenizer;
use crate::byte_level::bytes_to_string;
use crate::error::{Result, TokenizerError};
use crate::hf::HfTokenizer;

#[derive(Debug, Clone)]
pub struct TiktokenTokenizer {
    inner: HfTokenizer,
}

impl TiktokenTokenizer {
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { inner: HfTokenizer::from_file(path)? })
    }

    pub fn from_tiktoken_bpe(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TokenizerError::MissingFile(PathBuf::from(path)));
        }
        let raw = fs::read_to_string(path).map_err(|e| TokenizerError::io_at(path, e))?;
        let ranks = parse_tiktoken_bpe(&raw)?;
        let (vocab, merges) = ranks_to_vocab_and_merges(&ranks)?;
        let vocab_ahash: ahash::AHashMap<String, u32> = vocab.into_iter().collect();
        let model = BPE::builder()
            .vocab_and_merges(vocab_ahash, merges)
            .build()
            .map_err(TokenizerError::from)?;
        let mut tk = HfInner::new(model);
        tk.with_pre_tokenizer(Some(ByteLevelPre::default()));
        tk.with_decoder(Some(ByteLevelDecoder::default()));
        Ok(Self { inner: HfTokenizer::from_bytes(&serialize_tokenizer(&tk)?)? })
    }

    pub fn hf(&self) -> &HfTokenizer {
        &self.inner
    }

    pub fn into_hf(self) -> HfTokenizer {
        self.inner
    }
}

impl AsRef<HfTokenizer> for TiktokenTokenizer {
    fn as_ref(&self) -> &HfTokenizer {
        &self.inner
    }
}

fn parse_tiktoken_bpe(raw: &str) -> Result<HashMap<Vec<u8>, u32>> {
    let mut ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    for (lineno, line) in raw.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (encoded, rank) = line.split_once(' ').ok_or_else(|| {
            TokenizerError::invalid_arg(format!("line {} not in `base64 rank` form", lineno + 1))
        })?;
        let bytes = base64_decode(encoded.trim()).map_err(|e| {
            TokenizerError::invalid_arg(format!("line {}: bad base64: {}", lineno + 1, e))
        })?;
        let rank: u32 = rank.trim().parse().map_err(|e| {
            TokenizerError::invalid_arg(format!("line {}: bad rank: {}", lineno + 1, e))
        })?;
        ranks.insert(bytes, rank);
    }
    Ok(ranks)
}

fn ranks_to_vocab_and_merges(
    ranks: &HashMap<Vec<u8>, u32>,
) -> Result<(HashMap<String, u32>, Vec<(String, String)>)> {
    let mut sorted: Vec<(&Vec<u8>, &u32)> = ranks.iter().collect();
    sorted.sort_by_key(|(_, r)| **r);
    let mut vocab: HashMap<String, u32> = HashMap::with_capacity(ranks.len());
    let mut merges_indexed: Vec<(u32, String, String)> = Vec::new();
    for (bytes, rank) in &sorted {
        let token = bytes_to_string(bytes);
        vocab.insert(token, **rank);
        if bytes.len() <= 1 {
            continue;
        }
        let mut best: Option<(u32, usize)> = None;
        for split in 1..bytes.len() {
            let left = &bytes[..split];
            let right = &bytes[split..];
            let (Some(&lr), Some(&rr)) = (ranks.get(left), ranks.get(right)) else {
                continue;
            };
            if lr >= **rank || rr >= **rank {
                continue;
            }
            let score = lr.max(rr);
            if best.map(|(b, _)| score < b).unwrap_or(true) {
                best = Some((score, split));
            }
        }
        if let Some((_, split)) = best {
            let left = bytes_to_string(&bytes[..split]);
            let right = bytes_to_string(&bytes[split..]);
            merges_indexed.push((**rank, left, right));
        }
    }
    merges_indexed.sort_by_key(|(r, _, _)| *r);
    let merges = merges_indexed.into_iter().map(|(_, a, b)| (a, b)).collect();
    Ok((vocab, merges))
}

fn base64_decode(s: &str) -> std::result::Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::with_capacity(s.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for ch in s.chars() {
        if ch == '=' {
            break;
        }
        let value = match ch {
            'A'..='Z' => ch as u32 - 'A' as u32,
            'a'..='z' => ch as u32 - 'a' as u32 + 26,
            '0'..='9' => ch as u32 - '0' as u32 + 52,
            '+' | '-' => 62,
            '/' | '_' => 63,
            c if c.is_whitespace() => continue,
            _ => return Err(format!("invalid base64 char `{}`", ch)),
        };
        buf = (buf << 6) | value;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_basic() {
        assert_eq!(base64_decode("aGVsbG8=").unwrap(), b"hello");
        assert_eq!(base64_decode("YQ==").unwrap(), b"a");
        assert_eq!(base64_decode("YWI=").unwrap(), b"ab");
        assert_eq!(base64_decode("YWJj").unwrap(), b"abc");
    }

    #[test]
    fn ranks_roundtrip_minimal() {
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::new();
        ranks.insert(b"a".to_vec(), 0);
        ranks.insert(b"b".to_vec(), 1);
        ranks.insert(b"ab".to_vec(), 2);
        let (vocab, merges) = ranks_to_vocab_and_merges(&ranks).unwrap();
        assert_eq!(vocab.len(), 3);
        assert_eq!(merges.len(), 1);
        assert_eq!(merges[0], (bytes_to_string(b"a"), bytes_to_string(b"b")));
    }
}

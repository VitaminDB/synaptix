//! Текстовый BPE YuE2 — тот же, что у Qwen: `qwen.tiktoken` (151 643 обычных
//! токена) плюс 208 специальных.
//!
//! Это НЕ аудио-токенизатор: семантические токены модель выдаёт сама, они
//! живут выше по словарю ([`crate::protocol::CODEC_OFFSET`]).
//!
//! Реализация повторяет tiktoken дословно (разбиение паттерном Qwen, затем
//! слияние байтовых пар по рангам). Через `tokenizers`-BPE это не сделать
//! точно: ранги пришлось бы превращать в список слияний, а разбиение —
//! в ByteLevel-регэксп GPT-2, где цифры склеиваются (`\p{N}+`), тогда как у
//! Qwen каждая цифра — отдельный кусок.

use std::collections::HashMap;
use std::path::Path;

use unicode_normalization::UnicodeNormalization;

use crate::YueError;

/// Сколько обычных (не специальных) токенов в `qwen.tiktoken`.
pub const ORDINARY_TOKENS: usize = 151_643;

/// Паттерн предразбиения Qwen. Отличия от GPT-2, из-за которых нельзя взять
/// готовый ByteLevel: `\p{N}` (цифра по одной) и `[\r\n]*` в хвосте пунктуации.
const PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Имена специальных токенов в порядке их ID, начиная с `<|endoftext|>`.
fn special_names() -> Vec<String> {
    let mut names: Vec<String> = vec![
        "<|endoftext|>".into(),
        "<|im_start|>".into(),
        "<|im_end|>".into(),
        "<R>".into(),
        "<S>".into(),
        "<X>".into(),
        "<mask>".into(),
        "<sep>".into(),
    ];
    for i in 0..200 {
        names.push(format!("<extra_{i}>"));
    }
    // Два места из хвоста `extra_*` отданы границам партитуры.
    names[204] = "<abc>".into();
    names[205] = "</abc>".into();
    names
}

pub struct Yue2Tokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    /// ID → байты, для обычных токенов.
    decoder: Vec<Vec<u8>>,
    specials: Vec<String>,
    re: fancy_regex::Regex,
}

impl Yue2Tokenizer {
    /// Разобрать `qwen.tiktoken` (строки вида `<base64> <ранг>`).
    pub fn from_tiktoken_bytes(raw: &[u8]) -> Result<Self, YueError> {
        let text = std::str::from_utf8(raw)
            .map_err(|e| YueError::Config(format!("qwen.tiktoken: не UTF-8: {e}")))?;
        let mut ranks: HashMap<Vec<u8>, u32> = HashMap::with_capacity(ORDINARY_TOKENS);
        let mut decoder: Vec<Vec<u8>> = vec![Vec::new(); ORDINARY_TOKENS];
        for (lineno, line) in text.lines().enumerate() {
            if line.is_empty() {
                continue;
            }
            let (encoded, rank) = line.split_once(' ').ok_or_else(|| {
                YueError::Config(format!("qwen.tiktoken строка {}: не в форме `base64 ранг`", lineno + 1))
            })?;
            let bytes = base64_decode(encoded.trim())
                .map_err(|e| YueError::Config(format!("qwen.tiktoken строка {}: {e}", lineno + 1)))?;
            let rank: u32 = rank.trim().parse().map_err(|_| {
                YueError::Config(format!("qwen.tiktoken строка {}: ранг не число", lineno + 1))
            })?;
            if (rank as usize) < decoder.len() {
                decoder[rank as usize] = bytes.clone();
            }
            ranks.insert(bytes, rank);
        }
        if ranks.len() != ORDINARY_TOKENS {
            return Err(YueError::Config(format!(
                "ожидался родной qwen.tiktoken ({ORDINARY_TOKENS} обычных токенов), а в файле {}",
                ranks.len()
            )));
        }
        let re = fancy_regex::Regex::new(PATTERN)
            .map_err(|e| YueError::Config(format!("паттерн предразбиения: {e}")))?;
        Ok(Self { ranks, decoder, specials: special_names(), re })
    }

    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, YueError> {
        let raw = std::fs::read(path.as_ref())
            .map_err(|e| YueError::Load(format!("{}: {e}", path.as_ref().display())))?;
        Self::from_tiktoken_bytes(&raw)
    }

    /// Токенизировать обычный текст. Специальные последовательности внутри
    /// текста специальными токенами НЕ становятся — как `encode_ordinary`
    /// у tiktoken.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        // NFC — часть протокола: без неё «é» из двух кодпойнтов дало бы
        // другие токены, чем при обучении.
        let text: String = text.nfc().collect();
        let mut out: Vec<u32> = Vec::with_capacity(text.len() / 3 + 8);
        for m in self.re.find_iter(&text) {
            let Ok(m) = m else { continue };
            let piece = m.as_str().as_bytes();
            if let Some(&token) = self.ranks.get(piece) {
                out.push(token);
                continue;
            }
            byte_pair_encode(piece, &self.ranks, &mut out);
        }
        out
    }

    /// Обратно в текст. ID вне словаря (семантические токены, границы музыки)
    /// пропускаются — ровно как у эталона, который режет их по `n_vocab`.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 3);
        for &id in ids {
            let idx = id as usize;
            if idx < self.decoder.len() {
                bytes.extend_from_slice(&self.decoder[idx]);
            } else if let Some(name) = self.specials.get(idx - self.decoder.len()) {
                bytes.extend_from_slice(name.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    pub fn ordinary_tokens(&self) -> usize {
        self.decoder.len()
    }
}

/// Слияние байтовых пар по рангам — алгоритм tiktoken: на каждом шаге
/// схлопывается пара с наименьшим рангом.
fn byte_pair_encode(piece: &[u8], ranks: &HashMap<Vec<u8>, u32>, out: &mut Vec<u32>) {
    if piece.is_empty() {
        return;
    }
    if piece.len() == 1 {
        if let Some(&t) = ranks.get(piece) {
            out.push(t);
        }
        return;
    }
    // parts[i] = (смещение начала куска i, ранг склейки кусков i и i+1)
    let mut parts: Vec<(usize, u32)> = Vec::with_capacity(piece.len() + 1);
    let mut min_rank: (u32, usize) = (u32::MAX, usize::MAX);
    for i in 0..piece.len() - 1 {
        let rank = ranks.get(&piece[i..i + 2]).copied().unwrap_or(u32::MAX);
        if rank < min_rank.0 {
            min_rank = (rank, i);
        }
        parts.push((i, rank));
    }
    parts.push((piece.len() - 1, u32::MAX));
    parts.push((piece.len(), u32::MAX));

    let get_rank = |parts: &Vec<(usize, u32)>, i: usize| -> u32 {
        if i + 3 < parts.len() {
            ranks.get(&piece[parts[i].0..parts[i + 3].0]).copied().unwrap_or(u32::MAX)
        } else {
            u32::MAX
        }
    };

    while min_rank.0 != u32::MAX {
        let i = min_rank.1;
        if i > 0 {
            parts[i - 1].1 = get_rank(&parts, i - 1);
        }
        parts[i].1 = get_rank(&parts, i);
        parts.remove(i + 1);

        min_rank = (u32::MAX, usize::MAX);
        for (i, &(_, rank)) in parts[..parts.len() - 1].iter().enumerate() {
            if rank < min_rank.0 {
                min_rank = (rank, i);
            }
        }
    }
    for w in parts.windows(2) {
        if let Some(&t) = ranks.get(&piece[w[0].0..w[1].0]) {
            out.push(t);
        }
    }
}

fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
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
            _ => return Err(format!("плохой символ base64 `{ch}`")),
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

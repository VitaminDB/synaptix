//! Нативный CLIP BPE-токенайзер (slow `CLIPTokenizer` из HF transformers).
//!
//! SDXL поставляет только «медленные» файлы — `vocab.json` (token→id) и
//! `merges.txt` (ранги BPE-пар), без `tokenizer.json`. ByteLevel-BPE из
//! [`synaptix_tokenizer::BpeTokenizer`] для CLIP не годится: у CLIP свой
//! алгоритм — `</w>`-маркер конца слова + специфичный regex-сплит + GPT-2
//! byte→unicode маппинг. Реализуем его здесь, bit-exact к HF (проверяется
//! `tests/ref_tokenizer.rs` против дампнутых `input_ids`).
//!
//! Пайплайн: текст → `whitespace_clean` + lower → regex-сплит на «слова» →
//! каждое слово в UTF-8 байты → byte_encoder → BPE-слияния (последний символ
//! несёт `</w>`) → id через `vocab`. Финал: `[bos] + ids + [eos]`, паддинг
//! `eos`-ом до `max_len` (для CLIP pad_token == eos == `<|endoftext|>`).

use std::collections::HashMap;
use std::path::Path;

use regex::Regex;

use crate::FluxError;

/// GPT-2/CLIP byte→unicode таблица: обратимое отображение всех 256 байт в
/// печатные кодпоинты (чтобы BPE работал над «безопасными» символами).
fn bytes_to_unicode() -> Vec<(u8, char)> {
    let mut bs: Vec<u32> = Vec::new();
    bs.extend(b'!' as u32..=b'~' as u32);
    bs.extend(0xA1u32..=0xAC);
    bs.extend(0xAEu32..=0xFF);
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.into_iter()
        .zip(cs)
        .map(|(b, c)| (b as u8, char::from_u32(c).expect("valid codepoint")))
        .collect()
}

pub struct ClipTokenizer {
    vocab: HashMap<String, u32>,
    bpe_ranks: HashMap<(String, String), usize>,
    byte_encoder: HashMap<u8, char>,
    pat: Regex,
    bos: u32,
    eos: u32,
    pad: u32,
}

impl ClipTokenizer {
    /// Грузит токенайзер из директории `tokenizer*/` SDXL (vocab.json +
    /// merges.txt + tokenizer_config.json). pad-токен берётся из конфига:
    /// у `tokenizer/` (CLIP-L) это `<|endoftext|>` (49407), а у `tokenizer_2/`
    /// (bigG) — `!` (id 0). pad-эмбеддинги идут в кросс-аттеншн UNet, поэтому
    /// их правильность важна для bit-exact.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, FluxError> {
        let dir = dir.as_ref();
        let pad_token = Self::pad_token_from_config(dir);
        Self::from_files(dir.join("vocab.json"), dir.join("merges.txt"), pad_token.as_deref())
    }

    /// Читает `pad_token` из tokenizer_config.json (строка либо AddedToken-
    /// объект с полем `content`). Возвращает None → дефолт `<|endoftext|>`.
    fn pad_token_from_config(dir: &Path) -> Option<String> {
        let bytes = std::fs::read(dir.join("tokenizer_config.json")).ok()?;
        let cfg: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
        match cfg.get("pad_token")? {
            serde_json::Value::String(s) => Some(s.clone()),
            serde_json::Value::Object(o) => {
                o.get("content").and_then(|c| c.as_str()).map(str::to_string)
            }
            _ => None,
        }
    }

    /// `pad_token` = None → паддинг `<|endoftext|>` (дефолт CLIP).
    pub fn from_files(
        vocab_path: impl AsRef<Path>,
        merges_path: impl AsRef<Path>,
        pad_token: Option<&str>,
    ) -> Result<Self, FluxError> {
        let vocab_bytes = std::fs::read(vocab_path.as_ref())
            .map_err(|e| FluxError::Tokenizer(format!("vocab.json: {e}")))?;
        let vocab: HashMap<String, u32> = serde_json::from_slice(&vocab_bytes)
            .map_err(|e| FluxError::Tokenizer(format!("vocab.json parse: {e}")))?;

        let merges_text = std::fs::read_to_string(merges_path.as_ref())
            .map_err(|e| FluxError::Tokenizer(format!("merges.txt: {e}")))?;
        let mut bpe_ranks = HashMap::new();
        let mut rank = 0usize;
        for line in merges_text.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("#version") {
                continue;
            }
            let mut it = line.split_whitespace();
            match (it.next(), it.next()) {
                (Some(a), Some(b)) => {
                    bpe_ranks.insert((a.to_string(), b.to_string()), rank);
                    rank += 1;
                }
                _ => continue,
            }
        }

        let byte_encoder: HashMap<u8, char> = bytes_to_unicode().into_iter().collect();

        // CLIP-regex: спец-токены, английские контракции, буквы, цифры (по
        // одной), пунктуация. IGNORECASE как в HF.
        let pat = Regex::new(
            r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+",
        )
        .expect("valid clip regex");

        let id = |t: &str| -> Result<u32, FluxError> {
            vocab
                .get(t)
                .copied()
                .ok_or_else(|| FluxError::Tokenizer(format!("missing token {t}")))
        };
        let bos = id("<|startoftext|>")?;
        let eos = id("<|endoftext|>")?;
        let pad = match pad_token {
            Some(t) => id(t)?,
            None => eos,
        };

        Ok(Self { vocab, bpe_ranks, byte_encoder, pat, bos, eos, pad })
    }

    pub fn bos(&self) -> u32 {
        self.bos
    }
    pub fn eos(&self) -> u32 {
        self.eos
    }

    fn get_pairs(word: &[String]) -> Vec<(String, String)> {
        let mut pairs = Vec::with_capacity(word.len().saturating_sub(1));
        for w in word.windows(2) {
            let p = (w[0].clone(), w[1].clone());
            if !pairs.contains(&p) {
                pairs.push(p);
            }
        }
        pairs
    }

    /// BPE-слияния над byte-encoded словом; `token` — строка из
    /// byte→unicode символов. Возвращает sub-слова (части словаря CLIP).
    fn bpe(&self, token: &str) -> Vec<String> {
        let chars: Vec<char> = token.chars().collect();
        if chars.is_empty() {
            return Vec::new();
        }
        let mut word: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
        let last = word.len() - 1;
        word[last] = format!("{}</w>", word[last]);
        if word.len() == 1 {
            return word;
        }

        loop {
            let pairs = Self::get_pairs(&word);
            let mut best: Option<(usize, (String, String))> = None;
            for p in pairs {
                if let Some(&r) = self.bpe_ranks.get(&p) {
                    match &best {
                        Some((br, _)) if *br <= r => {}
                        _ => best = Some((r, p)),
                    }
                }
            }
            let (first, second) = match best {
                Some((_, p)) => p,
                None => break,
            };

            let mut new_word: Vec<String> = Vec::with_capacity(word.len());
            let mut i = 0;
            while i < word.len() {
                match word[i..].iter().position(|w| *w == first) {
                    Some(off) => {
                        let j = i + off;
                        new_word.extend_from_slice(&word[i..j]);
                        i = j;
                    }
                    None => {
                        new_word.extend_from_slice(&word[i..]);
                        break;
                    }
                }
                if word[i] == first && i + 1 < word.len() && word[i + 1] == second {
                    new_word.push(format!("{first}{second}"));
                    i += 2;
                } else {
                    new_word.push(word[i].clone());
                    i += 1;
                }
            }
            word = new_word;
            if word.len() == 1 {
                break;
            }
        }
        word
    }

    /// Голые BPE-id текста (без спец-токенов и паддинга).
    pub fn encode_raw(&self, text: &str) -> Vec<u32> {
        let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        let cleaned = cleaned.to_lowercase();
        let mut ids = Vec::new();
        for m in self.pat.find_iter(&cleaned) {
            let mut byte_encoded = String::new();
            for b in m.as_str().bytes() {
                byte_encoded.push(self.byte_encoder[&b]);
            }
            for piece in self.bpe(&byte_encoded) {
                if let Some(&id) = self.vocab.get(&piece) {
                    ids.push(id);
                }
            }
        }
        ids
    }

    /// Полная токенизация под CLIP-энкодер: `[bos] + ids + [eos]`, обрезка до
    /// `max_len` и паддинг `eos` (== pad для CLIP) до `max_len`.
    pub fn encode(&self, text: &str, max_len: usize) -> Vec<u32> {
        let mut ids = Vec::with_capacity(max_len);
        ids.push(self.bos);
        ids.extend(self.encode_raw(text));
        if ids.len() > max_len - 1 {
            ids.truncate(max_len - 1);
        }
        ids.push(self.eos);
        while ids.len() < max_len {
            ids.push(self.pad);
        }
        ids
    }
}

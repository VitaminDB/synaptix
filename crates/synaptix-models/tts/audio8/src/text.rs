//! Промпт как у `ArkttsProcessor` upstream: части кодируются по отдельности
//! (`add_special_tokens=False`) и склеиваются; текст чистится до одиночных пробелов.

use synaptix_tokenizer::hf::HfTokenizer;

use crate::{Audio8Error, Result};

pub struct Prompt {
    /// Токены до кодов образца голоса (или весь промпт без образца).
    pub prefix: Vec<u32>,
    /// Токены после кодов образца (пусто без образца).
    pub suffix: Vec<u32>,
}

pub struct TextTokenizer {
    hf: HfTokenizer,
}

/// `" ".join(text.strip().split())`.
pub fn clean_text(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn has_speaker_tag(s: &str) -> bool {
    let mut rest = s;
    while let Some(i) = rest.find("<|speaker:") {
        let tail = &rest[i + "<|speaker:".len()..];
        let digits = tail.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits > 0 && tail[digits..].starts_with("|>") {
            return true;
        }
        rest = tail;
    }
    false
}

impl TextTokenizer {
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        Ok(Self { hf: HfTokenizer::from_bytes(b).map_err(|e| Audio8Error::Tokenizer(e.to_string()))? })
    }

    pub fn encode(&self, s: &str) -> Result<Vec<u32>> {
        if s.is_empty() {
            return Ok(Vec::new());
        }
        let enc = self.hf.inner().encode(s, false).map_err(|e| Audio8Error::Tokenizer(e.to_string()))?;
        Ok(enc.get_ids().to_vec())
    }

    fn parts(&self, parts: &[&str]) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for p in parts {
            out.extend(self.encode(p)?);
        }
        Ok(out)
    }

    /// Промпт синтеза; `reference_text` — расшифровка образца (только с образцом).
    pub fn prompt(&self, text: &str, reference_text: Option<&str>) -> Result<Prompt> {
        let target = clean_text(text);
        if target.is_empty() {
            return Err(Audio8Error::Other("пустой текст".into()));
        }
        match reference_text {
            None => Ok(Prompt {
                prefix: self.parts(&[
                    "<|im_start|>system\n",
                    "convert the provided text to speech",
                    "<|im_end|>\n",
                    "<|im_start|>user\n",
                    &target,
                    "<|im_end|>\n",
                    "<|im_start|>assistant\n<|voice|>",
                ])?,
                suffix: Vec::new(),
            }),
            Some(r) => {
                let r = clean_text(r);
                if r.is_empty() {
                    return Err(Audio8Error::Other("нужна расшифровка образца голоса".into()));
                }
                let r = if has_speaker_tag(&r) { r } else { format!("<|speaker:0|>{r}") };
                Ok(Prompt {
                    prefix: self.parts(&[
                        "<|im_start|>system\n",
                        "convert the provided text to speech reference to the following:\n\nText:\n",
                        &r,
                        "\n\nSpeech:\n",
                    ])?,
                    suffix: self.parts(&[
                        "<|im_end|>\n",
                        "<|im_start|>user\n",
                        &target,
                        "<|im_end|>\n",
                        "<|im_start|>assistant\n<|voice|>",
                    ])?,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speaker_tag_detection() {
        assert!(has_speaker_tag("<|speaker:0|>hi"));
        assert!(has_speaker_tag("a <|speaker:12|> b"));
        assert!(!has_speaker_tag("<|speaker:|> b"));
        assert_eq!(clean_text("  a\n b\t\tc "), "a b c");
    }
}

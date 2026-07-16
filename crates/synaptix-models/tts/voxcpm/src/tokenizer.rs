use std::collections::HashSet;

use synaptix_tokenizer::hf::HfTokenizer;

use crate::VoxError;

pub const AUDIO_START: u32 = 101;
pub const AUDIO_END: u32 = 102;
pub const REF_AUDIO_START: u32 = 103;
pub const REF_AUDIO_END: u32 = 104;

pub struct TextTokenizer {
    hf: HfTokenizer,
    multichar: HashSet<String>,
    unk_id: u32,
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

impl TextTokenizer {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, VoxError> {
        let hf = HfTokenizer::from_bytes(bytes).map_err(|e| VoxError::Tokenizer(e.to_string()))?;
        let vocab = hf.inner().get_vocab(true);
        let multichar = vocab
            .keys()
            .filter(|t| {
                let n = t.chars().count();
                n >= 2 && t.chars().all(is_cjk)
            })
            .cloned()
            .collect();
        let unk_id = hf.inner().token_to_id("<unk>").unwrap_or(0);
        Ok(Self { hf, multichar, unk_id })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, VoxError> {
        let enc = self
            .hf
            .inner()
            .encode(text, false)
            .map_err(|e| VoxError::Tokenizer(e.to_string()))?;
        let tokens = enc.get_tokens();
        let ids = enc.get_ids();
        let mut out = Vec::with_capacity(tokens.len());
        for (tok, &id) in tokens.iter().zip(ids.iter()) {
            let clean: String = tok.replace('\u{2581}', "");
            if self.multichar.contains(&clean) {
                for ch in clean.chars() {
                    let s = ch.to_string();
                    out.push(self.hf.inner().token_to_id(&s).unwrap_or(self.unk_id));
                }
            } else {
                out.push(id);
            }
        }
        Ok(out)
    }

    pub fn encode_with_audio_start(&self, text: &str) -> Result<Vec<u32>, VoxError> {
        let mut ids = self.encode(text)?;
        ids.push(AUDIO_START);
        Ok(ids)
    }
}

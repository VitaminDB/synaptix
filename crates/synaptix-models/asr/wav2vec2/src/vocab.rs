//! Словарь CTC из HF `vocab.json` (`{"<pad>": 0, "a": 7, ...}`).

use std::collections::HashMap;

use crate::Wav2Vec2Error;

#[derive(Debug, Clone)]
pub struct CtcVocab {
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
}

impl CtcVocab {
    pub fn from_json_bytes(bytes: &[u8]) -> Result<Self, Wav2Vec2Error> {
        let map: HashMap<String, u32> =
            serde_json::from_slice(bytes).map_err(|e| Wav2Vec2Error::Vocab(e.to_string()))?;
        let n = map.values().copied().max().map_or(0, |m| m as usize + 1);
        let mut tokens = vec![String::new(); n];
        for (tok, &id) in &map {
            tokens[id as usize] = tok.clone();
        }
        Ok(Self { tokens, ids: map })
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn token(&self, id: u32) -> &str {
        self.tokens.get(id as usize).map_or("", |s| s.as_str())
    }

    pub fn id(&self, token: &str) -> Option<u32> {
        self.ids.get(token).copied()
    }

    pub fn tokens(&self) -> &[String] {
        &self.tokens
    }
}

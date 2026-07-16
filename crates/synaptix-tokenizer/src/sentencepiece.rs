use std::path::Path;

use crate::error::Result;
use crate::hf::HfTokenizer;

#[derive(Debug, Clone)]
pub struct SentencePieceTokenizer {
    inner: HfTokenizer,
}

impl SentencePieceTokenizer {
    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { inner: HfTokenizer::from_file(path)? })
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(Self { inner: HfTokenizer::from_bytes(bytes)? })
    }

    pub fn hf(&self) -> &HfTokenizer {
        &self.inner
    }

    pub fn into_hf(self) -> HfTokenizer {
        self.inner
    }
}

impl AsRef<HfTokenizer> for SentencePieceTokenizer {
    fn as_ref(&self) -> &HfTokenizer {
        &self.inner
    }
}

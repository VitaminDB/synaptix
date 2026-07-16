use std::path::{Path, PathBuf};

use tokenizers::models::unigram::Unigram;
use tokenizers::Tokenizer as HfInner;

use crate::bpe::serialize_tokenizer;
use crate::error::{Result, TokenizerError};
use crate::hf::HfTokenizer;

#[derive(Debug, Clone)]
pub struct UnigramTokenizer {
    inner: HfTokenizer,
}

impl UnigramTokenizer {
    pub fn from_vocab(vocab: Vec<(String, f64)>, unk_id: Option<usize>) -> Result<Self> {
        let model = Unigram::from(vocab, unk_id, false).map_err(TokenizerError::from)?;
        let tk = HfInner::new(model);
        Ok(Self { inner: HfTokenizer::from_bytes(&serialize_tokenizer(&tk)?)? })
    }

    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TokenizerError::MissingFile(PathBuf::from(path)));
        }
        Ok(Self { inner: HfTokenizer::from_file(path)? })
    }

    pub fn hf(&self) -> &HfTokenizer {
        &self.inner
    }

    pub fn into_hf(self) -> HfTokenizer {
        self.inner
    }
}

impl AsRef<HfTokenizer> for UnigramTokenizer {
    fn as_ref(&self) -> &HfTokenizer {
        &self.inner
    }
}

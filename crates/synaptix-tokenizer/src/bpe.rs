use std::path::{Path, PathBuf};

use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel as ByteLevelPre;
use tokenizers::processors::byte_level::ByteLevel as ByteLevelPost;
use tokenizers::Tokenizer as HfInner;

use crate::error::{Result, TokenizerError};
use crate::hf::HfTokenizer;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    inner: HfTokenizer,
}

impl BpeTokenizer {
    pub fn from_files(vocab: impl AsRef<Path>, merges: impl AsRef<Path>) -> Result<Self> {
        let vocab = vocab.as_ref();
        let merges = merges.as_ref();
        if !vocab.exists() {
            return Err(TokenizerError::MissingFile(PathBuf::from(vocab)));
        }
        if !merges.exists() {
            return Err(TokenizerError::MissingFile(PathBuf::from(merges)));
        }
        let model = BPE::from_file(
            vocab.to_str().ok_or_else(|| TokenizerError::invalid_arg("vocab path must be utf-8"))?,
            merges
                .to_str()
                .ok_or_else(|| TokenizerError::invalid_arg("merges path must be utf-8"))?,
        )
        .build()
        .map_err(TokenizerError::from)?;
        let mut tk = HfInner::new(model);
        tk.with_pre_tokenizer(Some(ByteLevelPre::default()));
        tk.with_decoder(Some(ByteLevelDecoder::default()));
        tk.with_post_processor(Some(ByteLevelPost::default()));
        Ok(Self { inner: HfTokenizer::from_bytes(&serialize_tokenizer(&tk)?)? })
    }

    pub fn from_tokenizer_json(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self { inner: HfTokenizer::from_file(path)? })
    }

    pub fn hf(&self) -> &HfTokenizer {
        &self.inner
    }

    pub fn into_hf(self) -> HfTokenizer {
        self.inner
    }
}

impl AsRef<HfTokenizer> for BpeTokenizer {
    fn as_ref(&self) -> &HfTokenizer {
        &self.inner
    }
}

pub(crate) fn serialize_tokenizer(tk: &HfInner) -> Result<Vec<u8>> {
    let json = tk.to_string(false).map_err(TokenizerError::from)?;
    Ok(json.into_bytes())
}

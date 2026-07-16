use std::path::{Path, PathBuf};

use tokenizers::decoders::wordpiece::WordPiece as WordPieceDecoder;
use tokenizers::models::wordpiece::WordPiece;
use tokenizers::normalizers::bert::BertNormalizer;
use tokenizers::pre_tokenizers::bert::BertPreTokenizer;
use tokenizers::Tokenizer as HfInner;

use crate::bpe::serialize_tokenizer;
use crate::error::{Result, TokenizerError};
use crate::hf::HfTokenizer;

#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    inner: HfTokenizer,
}

impl WordPieceTokenizer {
    pub fn from_vocab_file(vocab: impl AsRef<Path>, unk_token: &str) -> Result<Self> {
        let vocab = vocab.as_ref();
        if !vocab.exists() {
            return Err(TokenizerError::MissingFile(PathBuf::from(vocab)));
        }
        let model = WordPiece::from_file(
            vocab.to_str().ok_or_else(|| TokenizerError::invalid_arg("vocab path must be utf-8"))?,
        )
        .unk_token(unk_token.to_string())
        .build()
        .map_err(TokenizerError::from)?;
        let mut tk = HfInner::new(model);
        let _ = tk.with_normalizer(Some(BertNormalizer::default()));
        let _ = tk.with_pre_tokenizer(Some(BertPreTokenizer));
        let _ = tk.with_decoder(Some(WordPieceDecoder::default()));
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

impl AsRef<HfTokenizer> for WordPieceTokenizer {
    fn as_ref(&self) -> &HfTokenizer {
        &self.inner
    }
}

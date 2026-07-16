use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use tokenizers::tokenizer::{EncodeInput, InputSequence};

use crate::added_vocab::AddedVocab;
use crate::encoding::Encoding;
use crate::error::{Result, TokenizerError};
use crate::special_tokens::{SpecialTokenKind, SpecialTokens};
use crate::tokenizer::Tokenizer;

#[derive(Clone)]
pub struct HfTokenizer {
    inner: Arc<tokenizers::Tokenizer>,
    specials: SpecialTokens,
    added: AddedVocab,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("vocab_size", &self.inner.get_vocab_size(true))
            .field("specials", &self.specials)
            .field("added", &self.added)
            .finish()
    }
}

impl HfTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !path.exists() {
            return Err(TokenizerError::MissingFile(PathBuf::from(path)));
        }
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| TokenizerError::InvalidFile {
                path: PathBuf::from(path),
                message: e.to_string(),
            })?;
        Ok(Self::wrap(inner))
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)?;
        Ok(Self::wrap(inner))
    }

    pub fn from_pretrained(tokenizer_json: impl AsRef<Path>) -> Result<Self> {
        let path = tokenizer_json.as_ref();
        let bytes = fs::read(path).map_err(|e| TokenizerError::io_at(path, e))?;
        Self::from_bytes(&bytes)
    }

    fn wrap(inner: tokenizers::Tokenizer) -> Self {
        let mut hf = Self {
            inner: Arc::new(inner),
            specials: SpecialTokens::default(),
            added: AddedVocab::default(),
        };
        hf.sync_added_vocab();
        hf.autodetect_specials();
        hf
    }

    fn sync_added_vocab(&mut self) {
        // bulk-extend: один rebuild матчера (add() на каждый токен = O(N²)
        // DFA-построений — 13s на 6415 added-токенах Gemma).
        let mut vocab = AddedVocab::default();
        let tokens = self.inner.get_added_tokens_decoder().into_iter().map(|(id, hf_tok)| {
            crate::added_vocab::AddedToken {
                content: hf_tok.content.clone(),
                id,
                special: hf_tok.special,
                single_word: hf_tok.single_word,
                lstrip: hf_tok.lstrip,
                rstrip: hf_tok.rstrip,
                normalized: hf_tok.normalized,
            }
        });
        let _ = vocab.extend(tokens);
        self.added = vocab;
    }

    fn autodetect_specials(&mut self) {
        let mut specials = SpecialTokens::default();
        for tok in self.added.iter() {
            if !tok.special {
                continue;
            }
            if let Some(kind) = match_kind_by_content(&tok.content) {
                specials.set(kind, tok.content.clone(), tok.id);
            }
        }
        self.specials = specials;
    }

    pub fn set_special_token(&mut self, kind: SpecialTokenKind, content: &str) -> Result<()> {
        let id = self
            .inner
            .token_to_id(content)
            .ok_or_else(|| TokenizerError::UnknownToken { token: content.to_string() })?;
        self.specials.set(kind, content.to_string(), id);
        Ok(())
    }

    pub fn inner(&self) -> &tokenizers::Tokenizer {
        &self.inner
    }

    pub fn added_vocab(&self) -> &AddedVocab {
        &self.added
    }
}

fn match_kind_by_content(content: &str) -> Option<SpecialTokenKind> {
    match content {
        "<s>" | "<|begin_of_text|>" | "<|bos|>" | "<bos>" | "<|startoftext|>" => Some(SpecialTokenKind::Bos),
        "</s>" | "<|end_of_text|>" | "<|eos|>" | "<eos>" | "<|endoftext|>" => Some(SpecialTokenKind::Eos),
        "<pad>" | "<|pad|>" | "[PAD]" => Some(SpecialTokenKind::Pad),
        "<unk>" | "<|unk|>" | "[UNK]" => Some(SpecialTokenKind::Unk),
        "<sep>" | "[SEP]" => Some(SpecialTokenKind::Sep),
        "<cls>" | "[CLS]" => Some(SpecialTokenKind::Cls),
        "<mask>" | "[MASK]" => Some(SpecialTokenKind::Mask),
        "<|im_start|>" => Some(SpecialTokenKind::ImStart),
        "<|im_end|>" => Some(SpecialTokenKind::ImEnd),
        "<tool_call>" => Some(SpecialTokenKind::ToolCallStart),
        "</tool_call>" => Some(SpecialTokenKind::ToolCallEnd),
        "<tool_response>" => Some(SpecialTokenKind::ToolResponseStart),
        "</tool_response>" => Some(SpecialTokenKind::ToolResponseEnd),
        "<think>" => Some(SpecialTokenKind::ThinkStart),
        "</think>" => Some(SpecialTokenKind::ThinkEnd),
        _ => None,
    }
}

impl Tokenizer for HfTokenizer {
    fn encode(&self, input: &str, add_special_tokens: bool) -> Result<Encoding> {
        let enc = self
            .inner
            .encode(EncodeInput::Single(InputSequence::Raw(input.into())), add_special_tokens)?;
        Ok(Encoding::from_hf(&enc))
    }

    fn encode_pair(&self, a: &str, b: &str, add_special_tokens: bool) -> Result<Encoding> {
        let enc = self.inner.encode(
            EncodeInput::Dual(InputSequence::Raw(a.into()), InputSequence::Raw(b.into())),
            add_special_tokens,
        )?;
        Ok(Encoding::from_hf(&enc))
    }

    fn encode_batch(&self, inputs: &[String], add_special_tokens: bool) -> Result<Vec<Encoding>> {
        let mapped: Vec<EncodeInput<'_>> = inputs
            .iter()
            .map(|s| EncodeInput::Single(InputSequence::Raw(s.as_str().into())))
            .collect();
        let encs = self.inner.encode_batch(mapped, add_special_tokens)?;
        Ok(encs.iter().map(Encoding::from_hf).collect())
    }

    fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        Ok(self.inner.decode(ids, skip_special_tokens)?)
    }

    fn decode_batch(&self, batches: &[Vec<u32>], skip_special_tokens: bool) -> Result<Vec<String>> {
        let refs: Vec<&[u32]> = batches.iter().map(|v| v.as_slice()).collect();
        Ok(self.inner.decode_batch(&refs, skip_special_tokens)?)
    }

    fn vocab_size(&self, with_added: bool) -> usize {
        self.inner.get_vocab_size(with_added)
    }

    fn id_to_token(&self, id: u32) -> Option<String> {
        self.inner.id_to_token(id)
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        self.inner.token_to_id(token)
    }

    fn special_tokens(&self) -> &SpecialTokens {
        &self.specials
    }
}

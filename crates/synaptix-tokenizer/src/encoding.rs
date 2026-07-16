use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Encoding {
    pub ids: Vec<u32>,
    pub type_ids: Vec<u32>,
    pub tokens: Vec<String>,
    pub words: Vec<Option<u32>>,
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
    pub attention_mask: Vec<u32>,
    pub overflowing: Vec<Encoding>,
}

impl Encoding {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn from_hf(enc: &tokenizers::Encoding) -> Self {
        Self {
            ids: enc.get_ids().to_vec(),
            type_ids: enc.get_type_ids().to_vec(),
            tokens: enc.get_tokens().to_vec(),
            words: enc.get_word_ids().to_vec(),
            offsets: enc.get_offsets().to_vec(),
            special_tokens_mask: enc.get_special_tokens_mask().to_vec(),
            attention_mask: enc.get_attention_mask().to_vec(),
            overflowing: enc.get_overflowing().iter().map(Self::from_hf).collect(),
        }
    }

    pub fn truncate(&mut self, max_len: usize, stride: usize, direction: TruncationDirection) {
        if self.len() <= max_len {
            return;
        }
        let off = match direction {
            TruncationDirection::Right => 0,
            TruncationDirection::Left => self.len() - max_len,
        };
        let _ = stride;
        self.ids.drain(off + max_len..);
        self.ids.drain(..off);
        self.type_ids.drain(off + max_len..);
        self.type_ids.drain(..off);
        self.tokens.drain(off + max_len..);
        self.tokens.drain(..off);
        self.words.drain(off + max_len..);
        self.words.drain(..off);
        self.offsets.drain(off + max_len..);
        self.offsets.drain(..off);
        self.special_tokens_mask.drain(off + max_len..);
        self.special_tokens_mask.drain(..off);
        self.attention_mask.drain(off + max_len..);
        self.attention_mask.drain(..off);
    }

    pub fn pad(&mut self, target_len: usize, pad_id: u32, pad_type_id: u32, pad_token: &str, direction: PaddingDirection) {
        if self.len() >= target_len {
            return;
        }
        let pad_count = target_len - self.len();
        let make_pad = |coll: &mut Vec<u32>, value: u32| {
            let mut padding = vec![value; pad_count];
            match direction {
                PaddingDirection::Right => coll.append(&mut padding),
                PaddingDirection::Left => {
                    padding.append(coll);
                    *coll = padding;
                }
            }
        };
        make_pad(&mut self.ids, pad_id);
        make_pad(&mut self.type_ids, pad_type_id);
        make_pad(&mut self.special_tokens_mask, 1);
        let attn_pad = 0u32;
        let mut attn_padding = vec![attn_pad; pad_count];
        match direction {
            PaddingDirection::Right => self.attention_mask.append(&mut attn_padding),
            PaddingDirection::Left => {
                attn_padding.append(&mut self.attention_mask);
                self.attention_mask = attn_padding;
            }
        }
        let mut token_pad = vec![pad_token.to_string(); pad_count];
        let mut words_pad: Vec<Option<u32>> = vec![None; pad_count];
        let mut offsets_pad = vec![(0usize, 0usize); pad_count];
        match direction {
            PaddingDirection::Right => {
                self.tokens.append(&mut token_pad);
                self.words.append(&mut words_pad);
                self.offsets.append(&mut offsets_pad);
            }
            PaddingDirection::Left => {
                token_pad.append(&mut self.tokens);
                self.tokens = token_pad;
                words_pad.append(&mut self.words);
                self.words = words_pad;
                offsets_pad.append(&mut self.offsets);
                self.offsets = offsets_pad;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PaddingDirection {
    #[default]
    Right,
    Left,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TruncationDirection {
    #[default]
    Right,
    Left,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaddingStrategy {
    None,
    Longest { pad_id: u32, pad_token: String, direction: PaddingDirection },
    MaxLength { length: usize, pad_id: u32, pad_token: String, direction: PaddingDirection },
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TruncationStrategy {
    None,
    LongestFirst { max_length: usize, stride: usize, direction: TruncationDirection },
}

impl Default for TruncationStrategy {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Debug, Clone, Default)]
pub struct EncodeOptions {
    pub add_special_tokens: bool,
    pub padding: PaddingStrategy,
    pub truncation: TruncationStrategy,
}

impl EncodeOptions {
    pub fn new() -> Self {
        Self { add_special_tokens: true, ..Default::default() }
    }

    pub fn no_special(self) -> Self {
        Self { add_special_tokens: false, ..self }
    }

    pub fn with_padding(mut self, p: PaddingStrategy) -> Self {
        self.padding = p;
        self
    }

    pub fn with_truncation(mut self, t: TruncationStrategy) -> Self {
        self.truncation = t;
        self
    }
}

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpecialTokenKind {
    Bos,
    Eos,
    Pad,
    Unk,
    Sep,
    Cls,
    Mask,
    ImStart,
    ImEnd,
    ToolCallStart,
    ToolCallEnd,
    ToolResponseStart,
    ToolResponseEnd,
    ThinkStart,
    ThinkEnd,
}

impl SpecialTokenKind {
    pub fn canonical(self) -> &'static str {
        match self {
            Self::Bos => "<|bos|>",
            Self::Eos => "<|eos|>",
            Self::Pad => "<|pad|>",
            Self::Unk => "<|unk|>",
            Self::Sep => "<|sep|>",
            Self::Cls => "<|cls|>",
            Self::Mask => "<|mask|>",
            Self::ImStart => "<|im_start|>",
            Self::ImEnd => "<|im_end|>",
            Self::ToolCallStart => "<tool_call>",
            Self::ToolCallEnd => "</tool_call>",
            Self::ToolResponseStart => "<tool_response>",
            Self::ToolResponseEnd => "</tool_response>",
            Self::ThinkStart => "<think>",
            Self::ThinkEnd => "</think>",
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpecialTokens {
    by_kind: BTreeMap<SpecialTokenKind, SpecialTokenSlot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpecialTokenSlot {
    pub token: String,
    pub id: u32,
}

impl SpecialTokens {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, kind: SpecialTokenKind, token: impl Into<String>, id: u32) {
        self.by_kind.insert(kind, SpecialTokenSlot { token: token.into(), id });
    }

    pub fn remove(&mut self, kind: SpecialTokenKind) -> Option<SpecialTokenSlot> {
        self.by_kind.remove(&kind)
    }

    pub fn get(&self, kind: SpecialTokenKind) -> Option<&SpecialTokenSlot> {
        self.by_kind.get(&kind)
    }

    pub fn id_of(&self, kind: SpecialTokenKind) -> Option<u32> {
        self.by_kind.get(&kind).map(|s| s.id)
    }

    pub fn token_of(&self, kind: SpecialTokenKind) -> Option<&str> {
        self.by_kind.get(&kind).map(|s| s.token.as_str())
    }

    pub fn iter(&self) -> impl Iterator<Item = (SpecialTokenKind, &SpecialTokenSlot)> {
        self.by_kind.iter().map(|(k, v)| (*k, v))
    }

    pub fn contains(&self, kind: SpecialTokenKind) -> bool {
        self.by_kind.contains_key(&kind)
    }

    pub fn len(&self) -> usize {
        self.by_kind.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_kind.is_empty()
    }

    pub fn ids(&self) -> Vec<u32> {
        self.by_kind.values().map(|s| s.id).collect()
    }

    pub fn bos_id(&self) -> Option<u32> {
        self.id_of(SpecialTokenKind::Bos)
    }

    pub fn eos_id(&self) -> Option<u32> {
        self.id_of(SpecialTokenKind::Eos)
    }

    pub fn pad_id(&self) -> Option<u32> {
        self.id_of(SpecialTokenKind::Pad)
    }
}

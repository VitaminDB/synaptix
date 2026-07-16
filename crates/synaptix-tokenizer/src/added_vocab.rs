use std::collections::HashMap;

use aho_corasick::{AhoCorasick, AhoCorasickKind, MatchKind};
use serde::{Deserialize, Serialize};

use crate::error::{Result, TokenizerError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AddedToken {
    pub content: String,
    pub id: u32,
    pub special: bool,
    pub single_word: bool,
    pub lstrip: bool,
    pub rstrip: bool,
    pub normalized: bool,
}

impl AddedToken {
    pub fn new(content: impl Into<String>, id: u32) -> Self {
        Self {
            content: content.into(),
            id,
            special: false,
            single_word: false,
            lstrip: false,
            rstrip: false,
            normalized: false,
        }
    }

    pub fn special(mut self, value: bool) -> Self {
        self.special = value;
        self
    }
}

#[derive(Default)]
pub struct AddedVocab {
    tokens: Vec<AddedToken>,
    by_content: HashMap<String, u32>,
    by_id: HashMap<u32, usize>,
    matcher: Option<AhoCorasick>,
    pattern_to_idx: Vec<usize>,
}

impl std::fmt::Debug for AddedVocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AddedVocab")
            .field("tokens", &self.tokens)
            .field("by_content", &self.by_content.len())
            .field("by_id", &self.by_id.len())
            .field("matcher", &self.matcher.is_some())
            .field("pattern_to_idx", &self.pattern_to_idx.len())
            .finish()
    }
}

impl Clone for AddedVocab {
    fn clone(&self) -> Self {
        let mut v = AddedVocab {
            tokens: self.tokens.clone(),
            by_content: self.by_content.clone(),
            by_id: self.by_id.clone(),
            matcher: None,
            pattern_to_idx: Vec::new(),
        };
        v.rebuild_matcher();
        v
    }
}

impl AddedVocab {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, token: AddedToken) -> Result<()> {
        self.insert(token)?;
        self.rebuild_matcher();
        Ok(())
    }

    /// Вставка без перестройки матчера (см. [`AddedVocab::extend`]: rebuild —
    /// O(N·L) DFA-построение, на каждый add это O(N²); Gemma 6415 added-токенов
    /// давали ~13s загрузки токенизатора).
    fn insert(&mut self, token: AddedToken) -> Result<()> {
        if self.by_content.contains_key(&token.content) {
            return Err(TokenizerError::InvalidArgument(format!(
                "duplicate added token `{}`",
                token.content
            )));
        }
        if self.by_id.contains_key(&token.id) {
            return Err(TokenizerError::InvalidArgument(format!(
                "duplicate added token id {}",
                token.id
            )));
        }
        let idx = self.tokens.len();
        self.by_content.insert(token.content.clone(), token.id);
        self.by_id.insert(token.id, idx);
        self.tokens.push(token);
        Ok(())
    }

    /// Bulk-вставка: один rebuild матчера в конце (НЕ на каждый токен).
    pub fn extend<I: IntoIterator<Item = AddedToken>>(&mut self, iter: I) -> Result<()> {
        for t in iter {
            self.insert(t)?;
        }
        self.rebuild_matcher();
        Ok(())
    }

    fn rebuild_matcher(&mut self) {
        if self.tokens.is_empty() {
            self.matcher = None;
            self.pattern_to_idx.clear();
            return;
        }
        let mut indexed: Vec<(usize, &str)> = self
            .tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (i, t.content.as_str()))
            .collect();
        indexed.sort_by(|a, b| b.1.len().cmp(&a.1.len()).then(a.0.cmp(&b.0)));
        let patterns: Vec<&str> = indexed.iter().map(|(_, s)| *s).collect();
        self.pattern_to_idx = indexed.into_iter().map(|(i, _)| i).collect();
        self.matcher = AhoCorasick::builder()
            .kind(Some(AhoCorasickKind::DFA))
            .match_kind(MatchKind::LeftmostLongest)
            .build(patterns)
            .ok();
    }

    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.by_content.get(token).copied()
    }

    pub fn id_to_token(&self, id: u32) -> Option<&AddedToken> {
        self.by_id.get(&id).and_then(|i| self.tokens.get(*i))
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &AddedToken> {
        self.tokens.iter()
    }

    pub fn split<'a>(&'a self, text: &'a str) -> Vec<Segment<'a>> {
        let Some(matcher) = self.matcher.as_ref() else {
            if text.is_empty() {
                return Vec::new();
            }
            return vec![Segment::Text(text)];
        };
        let mut out: Vec<Segment<'a>> = Vec::new();
        let mut cursor = 0usize;
        for m in matcher.find_iter(text) {
            let pat_idx = m.pattern().as_usize();
            let tok_idx = self.pattern_to_idx[pat_idx];
            let token = &self.tokens[tok_idx];
            if m.start() > cursor {
                out.push(Segment::Text(&text[cursor..m.start()]));
            }
            out.push(Segment::Added { token, span: (m.start(), m.end()) });
            cursor = m.end();
        }
        if cursor < text.len() {
            out.push(Segment::Text(&text[cursor..]));
        }
        out
    }
}

#[derive(Debug, Clone)]
pub enum Segment<'a> {
    Text(&'a str),
    Added { token: &'a AddedToken, span: (usize, usize) },
}

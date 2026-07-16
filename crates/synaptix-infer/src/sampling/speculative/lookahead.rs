use super::ngram_lookup;

/// Lookahead-декодер: n-gram-драфт, ограниченный окном последних `window`
/// токенов (в отличие от [`super::draft_model::NgramDraftModel`], который смотрит
/// всю историю). Полезно, когда повторяющиеся паттерны локальны.
pub struct LookaheadDecoder {
    pub window: usize,
    pub n_gram: usize,
}

impl LookaheadDecoder {
    pub fn new(window: usize, n_gram: usize) -> Self { Self { window, n_gram } }

    /// Предложить до `n_draft` токенов n-gram-поиском по последним `window` токенам.
    pub fn propose(&self, tokens: &[u32], n_draft: usize) -> Vec<u32> {
        let start = tokens.len().saturating_sub(self.window);
        ngram_lookup(&tokens[start..], self.n_gram, n_draft)
    }
}

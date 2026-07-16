use super::ngram_lookup;

/// N-gram (prompt-lookup) драфт-модель. Не нейросеть: предлагает продолжение,
/// копируя то, что исторически следовало за текущим суффиксом контекста.
pub struct NgramDraftModel {
    /// Максимальная длина суффикса для поиска совпадения.
    pub n: usize,
    /// Размер словаря — нужен для синтетических `draft_logits`.
    pub vocab_size: usize,
    pub history: Vec<u32>,
}

impl NgramDraftModel {
    pub fn new(n: usize, vocab_size: usize) -> Self {
        Self { n, vocab_size, history: Vec::new() }
    }
    pub fn update(&mut self, token: u32) { self.history.push(token); }
}

impl super::DraftModel for NgramDraftModel {
    fn draft(&mut self, tokens: &[u32], n_draft: usize) -> crate::error::Result<Vec<u32>> {
        Ok(ngram_lookup(tokens, self.n, n_draft))
    }

    /// Для каждого предложенного токена — пик-распределение (logit ≈ 30 на токене,
    /// 0 на остальных): draft-вероятность ≈ 1, поэтому `verify_tokens` принимает
    /// токен ровно с целевой вероятностью `t_prob` (rejection sampling сводится к
    /// проверке таргета). Длина выхода == длине драфта.
    fn draft_logits(&mut self, tokens: &[u32], n_draft: usize) -> crate::error::Result<Vec<Vec<f32>>> {
        let drafted = ngram_lookup(tokens, self.n, n_draft);
        let mut out = Vec::with_capacity(drafted.len());
        for &t in &drafted {
            let mut row = vec![0.0f32; self.vocab_size];
            if (t as usize) < self.vocab_size {
                row[t as usize] = 30.0;
            }
            out.push(row);
        }
        Ok(out)
    }
}

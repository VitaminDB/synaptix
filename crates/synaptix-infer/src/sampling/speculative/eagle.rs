use super::{ngram_lookup, DraftModel};
use crate::error::Result;

/// EAGLE-style feature-level draft model: использует prompt-lookup как low-cost
/// «feature surrogate» и оборачивает результат в правдоподобные logits через
/// label-smoothing.
///
/// Полноценный EAGLE проецирует последний hidden state target-модели в малую
/// auto-regressive head (см. EAGLE-1/2/3 papers). Без реально подключённой
/// target-модели имеет смысл только функциональная замена: prompt-lookup +
/// confidence-аппроксимация. Когда появится host для feature-проектора, эту
/// реализацию можно подменить заглушкой → `Linear::forward` без изменений API.
pub struct EagleDraftModel {
    pub vocab_size: usize,
    pub max_ngram: usize,
    pub draft_confidence: f32,
}

impl EagleDraftModel {
    pub fn new(vocab_size: usize) -> Self {
        Self { vocab_size, max_ngram: 4, draft_confidence: 0.92 }
    }

    pub fn with_ngram(mut self, max_ngram: usize) -> Self {
        self.max_ngram = max_ngram.max(1);
        self
    }

    pub fn with_confidence(mut self, conf: f32) -> Self {
        self.draft_confidence = conf.clamp(0.0, 1.0);
        self
    }

    fn fallback_token(&self, tokens: &[u32]) -> u32 {
        tokens.last().copied().unwrap_or(0).min((self.vocab_size as u32).saturating_sub(1))
    }

    fn build_logits(&self, picked: u32) -> Vec<f32> {
        let conf = self.draft_confidence.clamp(1.0e-3, 1.0 - 1.0e-3);
        let bg = ((1.0 - conf) / (self.vocab_size.max(1) as f32 - 1.0)).max(1.0e-12);
        let mut row = vec![bg.ln(); self.vocab_size];
        if (picked as usize) < self.vocab_size {
            row[picked as usize] = conf.ln();
        }
        row
    }
}

impl DraftModel for EagleDraftModel {
    fn draft(&mut self, tokens: &[u32], n_draft: usize) -> Result<Vec<u32>> {
        if n_draft == 0 {
            return Ok(Vec::new());
        }
        let mut out = ngram_lookup(tokens, self.max_ngram, n_draft);
        if out.is_empty() {
            out.push(self.fallback_token(tokens));
        }
        if out.len() < n_draft {
            let last = *out.last().unwrap();
            for _ in out.len()..n_draft {
                out.push(last);
            }
        }
        for v in &mut out {
            if (*v as usize) >= self.vocab_size {
                *v = 0;
            }
        }
        Ok(out)
    }

    fn draft_logits(&mut self, tokens: &[u32], n_draft: usize) -> Result<Vec<Vec<f32>>> {
        let drafts = self.draft(tokens, n_draft)?;
        Ok(drafts.into_iter().map(|t| self.build_logits(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_uses_ngram_lookup_for_recurring_pattern() {
        let mut m = EagleDraftModel::new(64);
        let tokens = vec![3, 5, 7, 9, 3, 5, 7];
        let out = m.draft(&tokens, 3).unwrap();
        assert_eq!(out[0], 9);
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|&t| (t as usize) < m.vocab_size));
    }

    #[test]
    fn draft_falls_back_on_unique_suffix() {
        let mut m = EagleDraftModel::new(64);
        let tokens = vec![1, 2, 3];
        let out = m.draft(&tokens, 2).unwrap();
        assert_eq!(out.len(), 2);
        for &t in &out {
            assert!((t as usize) < m.vocab_size);
        }
    }

    #[test]
    fn draft_logits_picks_drafted_token_as_argmax() {
        let mut m = EagleDraftModel::new(32).with_confidence(0.9);
        let tokens = vec![3, 5, 7, 3, 5];
        let drafts = m.draft(&tokens, 2).unwrap();
        let logits = m.draft_logits(&tokens, 2).unwrap();
        assert_eq!(logits.len(), 2);
        for (row, &expected) in logits.iter().zip(drafts.iter()) {
            assert_eq!(row.len(), 32);
            let (argmax, _) = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap();
            assert_eq!(argmax as u32, expected);
            let max_l = row[argmax as usize];
            let other = row.iter().enumerate().filter(|(i, _)| *i != argmax).map(|(_, v)| *v)
                .fold(f32::NEG_INFINITY, f32::max);
            assert!(max_l > other, "drafted token must dominate (max={max_l} bg={other})");
        }
    }

    #[test]
    fn draft_zero_size_returns_empty() {
        let mut m = EagleDraftModel::new(32);
        let out = m.draft(&[1, 2, 3], 0).unwrap();
        assert!(out.is_empty());
        let lg = m.draft_logits(&[1, 2, 3], 0).unwrap();
        assert!(lg.is_empty());
    }

    #[test]
    fn draft_clamps_token_to_vocab() {
        let mut m = EagleDraftModel::new(4);
        let tokens = vec![100, 200, 100];
        let out = m.draft(&tokens, 2).unwrap();
        for &t in &out {
            assert!((t as usize) < m.vocab_size, "out-of-vocab draft={t}");
        }
    }

    #[test]
    fn confidence_extremes_clamped_safely() {
        let mut m = EagleDraftModel::new(8).with_confidence(0.0);
        let logits = m.draft_logits(&[1, 2, 3], 1).unwrap();
        for row in &logits {
            assert!(row.iter().all(|v| v.is_finite()));
        }
        let mut m2 = EagleDraftModel::new(8).with_confidence(1.0);
        let logits2 = m2.draft_logits(&[1, 2, 3], 1).unwrap();
        for row in &logits2 {
            assert!(row.iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn integrates_with_verify_tokens() {
        use super::super::verify_tokens;
        use synaptix_ops::rng::Philox4x32;
        let mut m = EagleDraftModel::new(8).with_confidence(0.95);
        let tokens = vec![1, 2, 3, 1, 2];
        let drafts = m.draft(&tokens, 3).unwrap();
        let dl = m.draft_logits(&tokens, 3).unwrap();
        let mut target = vec![vec![0.1f32; 8]; 3];
        for (row, &t) in target.iter_mut().zip(drafts.iter()) {
            row[t as usize] = 5.0;
        }
        let mut rng = Philox4x32::new(0xCAFE);
        let out = verify_tokens(&drafts, &dl, &target, &mut rng);
        assert!(!out.accepted.is_empty(), "perfect-match target should accept");
    }
}

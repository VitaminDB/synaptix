pub mod draft_model;
pub mod eagle;
pub mod lookahead;
pub mod medusa;
pub mod self_spec;
pub mod tree_attn;

use synaptix_ops::rng::Philox4x32;

pub trait DraftModel: Send {
    fn draft(&mut self, tokens: &[u32], n_draft: usize) -> crate::error::Result<Vec<u32>>;
    fn draft_logits(&mut self, tokens: &[u32], n_draft: usize) -> crate::error::Result<Vec<Vec<f32>>>;
}

pub struct SpeculativeOutput {
    pub accepted: Vec<u32>,
    pub rejected_at: Option<usize>,
    pub bonus_token: Option<u32>,
}

pub fn verify_tokens(
    draft_tokens: &[u32],
    draft_logits: &[Vec<f32>],
    target_logits: &[Vec<f32>],
    rng: &mut Philox4x32,
) -> SpeculativeOutput {
    let mut accepted = Vec::new();
    for (i, &tok) in draft_tokens.iter().enumerate() {
        let t_prob = softmax_prob(&target_logits[i], tok as usize);
        let d_prob = softmax_prob(&draft_logits[i], tok as usize).max(1e-10);
        let accept_prob = (t_prob / d_prob).min(1.0);
        let u = rng.next_f32_uniform();
        if u < accept_prob {
            accepted.push(tok);
        } else {
            return SpeculativeOutput { accepted, rejected_at: Some(i), bonus_token: None };
        }
    }
    SpeculativeOutput { accepted, rejected_at: None, bonus_token: None }
}

fn softmax_prob(logits: &[f32], idx: usize) -> f32 {
    if idx >= logits.len() { return 0.0; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    (logits[idx] - max).exp() / exp_sum.max(1e-10)
}

/// Prompt-lookup (n-gram) драфт без отдельной модели. Берёт суффикс из последних
/// `m` токенов (`m` от `max_ngram` вниз до 1) и ищет его более раннее вхождение
/// в `tokens`; предлагает до `n_draft` токенов, которые исторически следовали за
/// этим вхождением. Берётся самое длинное совпадение, среди равных — правейшее
/// (самое свежее). Полностью детерминированно (см. суффикс-поиск в
/// `synaptix-ops::sampling_ops::dry`). Пусто, если совпадения нет.
pub fn ngram_lookup(tokens: &[u32], max_ngram: usize, n_draft: usize) -> Vec<u32> {
    let len = tokens.len();
    if len < 2 || n_draft == 0 || max_ngram == 0 {
        return Vec::new();
    }
    let max_m = max_ngram.min(len - 1);
    for m in (1..=max_m).rev() {
        let suffix = &tokens[len - m..];
        let mut best: Option<usize> = None;
        // Вхождения, начинающиеся в [0, len-m-1] — за каждым есть продолжение.
        for i in 0..=(len - m - 1) {
            if &tokens[i..i + m] == suffix {
                best = Some(i); // правейшее побеждает
            }
        }
        if let Some(i) = best {
            let start = i + m;
            let end = (start + n_draft).min(len);
            if start < end {
                return tokens[start..end].to_vec();
            }
        }
    }
    Vec::new()
}

use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct TopPProcessor {
    pub p: f32,
}

impl LogitProcessor for TopPProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        if self.p >= 1.0 {
            return Ok(());
        }
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = probs.iter().sum();

        // Сортируем только КАНДИДАТОВ (finite logits). После top_k их ≤k (=40 по
        // умолчанию в чате) → сортировка O(k log k) вместо O(V log V) по 248K.
        // Без top_k (top_k=0) — все finite, поведение прежнее. Нормировка не нужна:
        // -inf logits дают prob=0, sum уже = сумме finite, порог = p*sum.
        let mut indexed: Vec<(usize, f32)> = probs
            .iter()
            .copied()
            .enumerate()
            .filter(|&(i, _)| logits[i].is_finite())
            .collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let thresh = self.p * sum;

        let mut cumsum = 0.0f32;
        let mut cutoff = indexed.len();
        for (pos, (_, prob)) in indexed.iter().enumerate() {
            if cumsum > thresh {
                cutoff = pos;
                break;
            }
            cumsum += prob;
        }

        for i in cutoff..indexed.len() {
            logits[indexed[i].0] = f32::NEG_INFINITY;
        }
        Ok(())
    }
}

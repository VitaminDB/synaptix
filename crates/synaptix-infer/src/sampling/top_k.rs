use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct TopKProcessor {
    pub k: usize,
}

impl LogitProcessor for TopKProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        if self.k == 0 || self.k >= logits.len() {
            return Ok(());
        }
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        // Частичная выборка top-k за O(n) (вместо полной сортировки O(n log n) по
        // всему словарю — на 248K было ~3-4ms/токен). После select_nth первые k
        // элементов — наибольшие, остальное гасим в -inf.
        indexed.select_nth_unstable_by(self.k - 1, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        for i in self.k..indexed.len() {
            logits[indexed[i].0] = f32::NEG_INFINITY;
        }
        Ok(())
    }
}

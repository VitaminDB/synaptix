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
        // Порог — k-е по величине значение (select_nth по копии значений, O(n)); гасятся
        // строго меньшие. Равные порогу сверх k остаются (как у HF: `logits < kth`).
        // Без вектора пар (индекс, значение) на весь словарь — вдвое меньше памяти и времени.
        let mut vals: Vec<f32> = logits.iter().map(|&v| if v.is_nan() { f32::NEG_INFINITY } else { v }).collect();
        let (_, kth, _) = vals.select_nth_unstable_by(self.k - 1, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let kth = *kth;
        for v in logits.iter_mut() {
            if !(*v >= kth) {
                *v = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

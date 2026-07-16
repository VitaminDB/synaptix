use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct ContrastiveProcessor {
    pub alpha: f32,
    pub k: usize,
}

impl ContrastiveProcessor {
    pub fn new(alpha: f32, k: usize) -> Self { Self { alpha, k } }
}

impl LogitProcessor for ContrastiveProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        if self.k > 0 && self.k < logits.len() {
            let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for i in self.k..indexed.len() {
                logits[indexed[i].0] = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

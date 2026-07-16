use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct MinPProcessor {
    pub min_p: f32,
}

impl LogitProcessor for MinPProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        let max_prob = probs.iter().cloned().fold(0.0f32, f32::max);
        let threshold = self.min_p * max_prob;

        for (logit, &prob) in logits.iter_mut().zip(probs.iter()) {
            if prob < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

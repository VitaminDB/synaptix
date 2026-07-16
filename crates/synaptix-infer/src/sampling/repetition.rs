use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct RepetitionPenaltyProcessor {
    pub penalty: f32,
}

impl LogitProcessor for RepetitionPenaltyProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, ctx: &ProcessorContext) -> Result<()> {
        for &token_id in &ctx.input_ids {
            let idx = token_id as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= self.penalty;
                } else {
                    logits[idx] *= self.penalty;
                }
            }
        }
        Ok(())
    }
}

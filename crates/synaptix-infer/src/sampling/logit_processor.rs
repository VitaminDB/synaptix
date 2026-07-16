use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct TemperatureProcessor {
    pub temperature: f32,
}

impl LogitProcessor for TemperatureProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        if self.temperature <= 0.0 || self.temperature == 1.0 {
            return Ok(());
        }
        for logit in logits.iter_mut() {
            *logit /= self.temperature;
        }
        Ok(())
    }
}

pub struct FrequencyPenaltyProcessor {
    pub alpha: f32,
}

impl LogitProcessor for FrequencyPenaltyProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, ctx: &ProcessorContext) -> Result<()> {
        let mut counts = vec![0usize; logits.len()];
        for &token_id in &ctx.input_ids {
            let idx = token_id as usize;
            if idx < counts.len() {
                counts[idx] += 1;
            }
        }
        for (i, &count) in counts.iter().enumerate() {
            if count > 0 {
                logits[i] -= self.alpha * count as f32;
            }
        }
        Ok(())
    }
}

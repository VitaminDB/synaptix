use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct GrammarMaskProcessor {
    pub allowed_tokens: Vec<u32>,
}

impl GrammarMaskProcessor {
    pub fn new(allowed: Vec<u32>) -> Self { Self { allowed_tokens: allowed } }
    pub fn update_allowed(&mut self, allowed: Vec<u32>) { self.allowed_tokens = allowed; }
}

impl LogitProcessor for GrammarMaskProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        let allowed: std::collections::HashSet<u32> = self.allowed_tokens.iter().copied().collect();
        for (i, logit) in logits.iter_mut().enumerate() {
            if !allowed.contains(&(i as u32)) {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

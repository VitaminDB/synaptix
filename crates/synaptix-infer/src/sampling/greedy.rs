use synaptix_ops::rng::Philox4x32;
use crate::error::{InferError, Result};
use super::Sampler;

pub struct GreedySampler;

impl Sampler for GreedySampler {
    fn sample(&mut self, logits: &[f32], _rng: &mut Philox4x32) -> Result<u32> {
        logits.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .ok_or_else(|| InferError::Sampling("empty logits".into()))
    }
}

use synaptix_ops::rng::Philox4x32;
use crate::error::Result;
use super::Sampler;

pub struct MultinomialSampler;

impl Sampler for MultinomialSampler {
    fn sample(&mut self, logits: &[f32], rng: &mut Philox4x32) -> Result<u32> {
        let probs = softmax(logits);
        let u = next_f32(rng);
        let mut cumsum = 0.0f32;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u < cumsum {
                return Ok(i as u32);
            }
        }
        Ok((probs.len() - 1) as u32)
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    for e in exps.iter_mut() { *e /= sum; }
    exps
}

fn next_f32(rng: &mut Philox4x32) -> f32 {
    rng.next_u32() as f64 as f32 / u32::MAX as f32
}

use synaptix_ops::rng::Philox4x32;
use crate::error::{InferError, Result};
use super::Sampler;

pub struct MirostatV2Sampler {
    pub tau: f32,
    pub eta: f32,
    pub mu: f32,
}

impl MirostatV2Sampler {
    pub fn new(tau: f32, eta: f32) -> Self {
        Self { tau, eta, mu: tau * 2.0 }
    }
}

impl Sampler for MirostatV2Sampler {
    fn sample(&mut self, logits: &[f32], rng: &mut Philox4x32) -> Result<u32> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut kept: Vec<(usize, f32)> = Vec::new();
        let mut cum_surprise = 0.0f32;
        for (idx, prob) in indexed {
            let surprise = if prob > 0.0 { -prob.log2() } else { f32::INFINITY };
            if cum_surprise + surprise > self.mu && !kept.is_empty() {
                break;
            }
            cum_surprise += surprise;
            kept.push((idx, prob));
        }

        if kept.is_empty() {
            return Err(InferError::Sampling("mirostat: no tokens kept".into()));
        }

        let kept_sum: f32 = kept.iter().map(|&(_, p)| p).sum();
        let u = rng.next_f32_uniform();
        let mut cumsum = 0.0f32;
        let mut sampled_idx = kept[0].0;
        let mut sampled_prob = kept[0].1;
        for &(idx, prob) in &kept {
            cumsum += prob / kept_sum;
            if u < cumsum {
                sampled_idx = idx;
                sampled_prob = prob;
                break;
            }
        }

        let surprise = if sampled_prob > 0.0 { -sampled_prob.log2() } else { f32::INFINITY };
        self.mu -= self.eta * (surprise - self.tau);

        Ok(sampled_idx as u32)
    }
}

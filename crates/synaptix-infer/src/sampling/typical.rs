use crate::error::Result;
use super::{LogitProcessor, ProcessorContext};

pub struct TypicalProcessor {
    pub mass: f32,
}

impl LogitProcessor for TypicalProcessor {
    fn process(&mut self, logits: &mut Vec<f32>, _ctx: &ProcessorContext) -> Result<()> {
        let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();

        let entropy: f32 = probs.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum();

        let mut deviations: Vec<(usize, f32)> = probs.iter().enumerate()
            .map(|(i, &p)| {
                let log_p = if p > 0.0 { p.ln() } else { f32::NEG_INFINITY };
                (i, (log_p + entropy).abs())
            })
            .collect();
        deviations.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut cumsum = 0.0f32;
        let mut cutoff = deviations.len();
        for (pos, &(idx, _)) in deviations.iter().enumerate() {
            cumsum += probs[idx];
            if cumsum >= self.mass {
                cutoff = pos + 1;
                break;
            }
        }

        let kept: std::collections::HashSet<usize> = deviations[..cutoff].iter().map(|&(i, _)| i).collect();
        for (i, logit) in logits.iter_mut().enumerate() {
            if !kept.contains(&i) {
                *logit = f32::NEG_INFINITY;
            }
        }
        Ok(())
    }
}

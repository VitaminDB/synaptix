#[derive(Clone)]
pub struct Beam {
    pub tokens: Vec<u32>,
    pub score: f32,
}

pub struct BeamSearchSampler {
    pub num_beams: usize,
    pub length_penalty: f32,
}

impl BeamSearchSampler {
    pub fn new(num_beams: usize) -> Self {
        Self { num_beams, length_penalty: 1.0 }
    }

    pub fn expand_beams(&self, beams: &[Beam], logits_per_beam: &[Vec<f32>]) -> Vec<Beam> {
        let mut candidates: Vec<Beam> = Vec::new();

        for (beam, logits) in beams.iter().zip(logits_per_beam.iter()) {
            let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();

            let mut indexed: Vec<(usize, f32)> = exps.iter()
                .enumerate()
                .map(|(i, &e)| (i, e / sum))
                .collect();
            indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (token_id, prob) in indexed.iter().take(self.num_beams) {
                let log_prob = prob.max(1e-10).ln();
                let len = (beam.tokens.len() + 1) as f32;
                let penalized_score = (beam.score * beam.tokens.len() as f32 + log_prob)
                    / len.powf(self.length_penalty);
                let mut new_tokens = beam.tokens.clone();
                new_tokens.push(*token_id as u32);
                candidates.push(Beam { tokens: new_tokens, score: penalized_score });
            }
        }

        candidates.sort_unstable_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(self.num_beams);
        candidates
    }
}

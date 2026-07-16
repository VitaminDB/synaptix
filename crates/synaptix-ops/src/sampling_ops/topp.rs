pub fn apply_topp(logits: &mut Vec<f32>, p: f32) {
    if p >= 1.0 { return; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let mut probs: Vec<(usize, f32)> = logits.iter().map(|&x| (x - max).exp() / exp_sum).enumerate().collect();
    probs.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0_f32;
    for (idx, prob) in &probs {
        cum += prob;
        if cum > p { logits[*idx] = f32::NEG_INFINITY; }
    }
}

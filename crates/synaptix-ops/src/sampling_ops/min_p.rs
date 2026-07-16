pub fn apply_min_p(logits: &mut Vec<f32>, min_p: f32) {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let top_prob = (max - max).exp() / exp_sum;
    let threshold = min_p * top_prob;
    for (i, &l) in logits.clone().iter().enumerate() {
        let prob = (l - max).exp() / exp_sum;
        if prob < threshold { logits[i] = f32::NEG_INFINITY; }
    }
}

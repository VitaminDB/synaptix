pub fn apply_typical(logits: &mut Vec<f32>, mass: f32) {
    if mass >= 1.0 { return; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let probs: Vec<f32> = logits.iter().map(|&x| (x - max).exp() / exp_sum).collect();
    let h: f32 = -probs.iter().filter(|&&p| p > 0.0).map(|&p| p * p.ln()).sum::<f32>();
    let mut scored: Vec<(usize, f32)> = probs.iter().enumerate()
        .map(|(i, &p)| (i, if p > 0.0 { (p.ln().abs() - h).abs() } else { f32::MAX }))
        .collect();
    scored.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0_f32;
    for (idx, _) in &scored {
        cum += probs[*idx];
        if cum > mass { logits[*idx] = f32::NEG_INFINITY; }
    }
}

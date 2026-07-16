pub fn apply_topk(logits: &mut Vec<f32>, k: usize) {
    if k == 0 || k >= logits.len() { return; }
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for i in k..indexed.len() { logits[indexed[i].0] = f32::NEG_INFINITY; }
}

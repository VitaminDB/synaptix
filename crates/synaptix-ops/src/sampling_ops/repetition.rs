pub fn apply_repetition_penalty(logits: &mut Vec<f32>, input_ids: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 { return; }
    for &id in input_ids {
        if (id as usize) < logits.len() {
            if logits[id as usize] > 0.0 { logits[id as usize] /= penalty; }
            else { logits[id as usize] *= penalty; }
        }
    }
}

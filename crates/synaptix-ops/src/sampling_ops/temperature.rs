pub fn apply_temperature(logits: &mut Vec<f32>, temperature: f32) {
    if temperature <= 0.0 || temperature == 1.0 { return; }
    for l in logits.iter_mut() { *l /= temperature; }
}

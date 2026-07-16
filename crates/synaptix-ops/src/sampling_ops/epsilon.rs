/// Epsilon sampling: маскирует (в `-inf`) токены, чья softmax-вероятность < `epsilon`.
/// Если под порог попадают все токены, оставляется argmax (защита от пустого распределения).
pub fn apply_epsilon(logits: &mut Vec<f32>, epsilon: f32) {
    if epsilon <= 0.0 || logits.is_empty() {
        return;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let exp_sum: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    if exp_sum <= 0.0 {
        return;
    }
    let argmax = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap();

    let mut kept = 0usize;
    for x in logits.iter_mut() {
        let prob = (*x - max).exp() / exp_sum;
        if prob < epsilon {
            *x = f32::NEG_INFINITY;
        } else {
            kept += 1;
        }
    }
    if kept == 0 {
        // вернуть наиболее вероятный токен
        // (его логит до маскирования был max → восстановим)
        logits[argmax] = max;
    }
}

/// XTC (Exclude Top Choices): среди токенов с softmax-вероятностью ≥ `threshold`
/// маскирует (в `-inf`) все, кроме наименее вероятного — оставляя «хвост» и один
/// из топовых. `probability` — стохастический гейт (выполняется выше по стеку);
/// здесь маскирование применяется при `probability > 0` (для детерминизма
/// тестов передают 1.0; `probability ≤ 0` — no-op).
pub fn apply_xtc(logits: &mut Vec<f32>, probability: f32, threshold: f32) {
    if probability <= 0.0 || logits.is_empty() {
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
    // индексы токенов с вероятностью >= threshold
    let mut above: Vec<usize> = (0..logits.len())
        .filter(|&i| (logits[i] - max).exp() / exp_sum >= threshold)
        .collect();
    if above.len() <= 1 {
        return; // нечего исключать
    }
    // отсортировать по вероятности по убыванию; оставить наименее вероятный из топа
    above.sort_unstable_by(|&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    for &idx in &above[..above.len() - 1] {
        logits[idx] = f32::NEG_INFINITY;
    }
}
